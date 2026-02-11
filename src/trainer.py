
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from datasets import Dataset
import wandb

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ["qkv_proj", "o_proj"]
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_OUTPUT_DIR = "./checkpoints"


def detect_device() -> str:
    """Detect the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FinancialLLMTrainer:

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.device_type = detect_device()
        logger.info(f"Detected device: {self.device_type}")

        # Extract configuration with defaults
        self.model_name = self.config.get("model_name", DEFAULT_MODEL_NAME)
        self.lora_rank = self.config.get("lora_rank", DEFAULT_LORA_RANK)
        self.lora_alpha = self.config.get("lora_alpha", DEFAULT_LORA_ALPHA)
        self.lora_dropout = self.config.get("lora_dropout", DEFAULT_LORA_DROPOUT)
        self.target_modules = self.config.get("target_modules", DEFAULT_TARGET_MODULES)
        self.learning_rate = self.config.get("learning_rate", DEFAULT_LEARNING_RATE)
        self.num_epochs = self.config.get("num_epochs", DEFAULT_NUM_EPOCHS)
        self.per_device_train_batch_size = self.config.get(
            "per_device_train_batch_size", DEFAULT_BATCH_SIZE
        )
        self.per_device_eval_batch_size = self.config.get(
            "per_device_eval_batch_size", DEFAULT_BATCH_SIZE
        )
        self.output_dir = self.config.get("output_dir", DEFAULT_OUTPUT_DIR)

        # W&B settings
        self.wandb_project = self.config.get("wandb_project", "financial-llm-advisor")
        self.wandb_enabled = self.config.get("wandb_enabled", True)

        # Training settings
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        self.warmup_steps = self.config.get("warmup_steps", 500)
        self.logging_steps = self.config.get("logging_steps", 100)
        self.eval_steps = self.config.get("eval_steps", 500)
        self.save_steps = self.config.get("save_steps", 500)
        self.save_total_limit = self.config.get("save_total_limit", 3)
        self.gradient_checkpointing = self.config.get("gradient_checkpointing", True)
        self.fp16 = self.config.get("fp16", False)
        self.bf16 = self.config.get("bf16", False)
        self.optim = self.config.get("optim", "adamw_8bit")
        self.seed = self.config.get("seed", 42)

        # Quantization settings (config values = CUDA-optimal defaults)
        self.load_in_8bit = self.config.get("load_in_8bit", True)
        self.load_in_4bit = self.config.get("load_in_4bit", False)

        # Auto-adjust settings for the detected device
        self._apply_device_overrides()

        # Initialize state
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.trainer: Optional[Trainer] = None
        self.peft_config: Optional[LoraConfig] = None

    def _apply_device_overrides(self) -> None:
        """Override config settings that are incompatible with the detected device."""
        if self.device_type == "cuda":
            return

        overrides: List[str] = []

        # bitsandbytes quantization is CUDA-only
        if self.load_in_8bit or self.load_in_4bit:
            self.load_in_8bit = False
            self.load_in_4bit = False
            overrides.append("quantization disabled (bitsandbytes requires CUDA)")

        # adamw_8bit optimizer is CUDA-only
        if self.optim == "adamw_8bit":
            self.optim = "adamw_torch"
            overrides.append("optimizer adamw_8bit -> adamw_torch")

        # fp16 is not well-supported on MPS; prefer no mixed precision or bf16
        if self.device_type == "mps" and self.fp16:
            self.fp16 = False
            overrides.append("fp16 disabled (not stable on MPS)")

        # Reduce batch size to compensate for no quantization, keep effective batch the same
        config_batch = self.config.get("per_device_train_batch_size", DEFAULT_BATCH_SIZE)
        if config_batch > 4:
            factor = config_batch // 4
            self.per_device_train_batch_size = 4
            self.per_device_eval_batch_size = 4
            self.gradient_accumulation_steps = max(
                self.gradient_accumulation_steps * factor, factor
            )
            overrides.append(
                f"batch {config_batch} -> 4 (grad_accum={self.gradient_accumulation_steps}) "
                f"for memory without quantization"
            )

        if overrides:
            logger.info(f"Device overrides for {self.device_type}:")
            for o in overrides:
                logger.info(f"  - {o}")

    def load_model(self, model_name: Optional[str] = None) -> PreTrainedModel:

        model_name = model_name or self.model_name
        logger.info(f"Loading model: {model_name}")

        # Configure quantization
        if self.load_in_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif self.load_in_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        # Load model
        # Temporarily disable _init_weights to avoid dtype errors with quantized weights.
        # All weights come from pretrained checkpoint so re-initialization is unnecessary.
        from transformers import modeling_utils
        _orig = modeling_utils.PreTrainedModel._initialize_missing_keys
        modeling_utils.PreTrainedModel._initialize_missing_keys = lambda self, *a, **kw: None
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        finally:
            modeling_utils.PreTrainedModel._initialize_missing_keys = _orig

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare model for k-bit training
        if quantization_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )

        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        return self.model

    def setup_lora(
        self,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
    ) -> PeftModel:

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Use provided values or defaults
        rank = rank or self.lora_rank
        alpha = alpha or self.lora_alpha
        dropout = dropout or self.lora_dropout
        target_modules = target_modules or self.target_modules

        logger.info(f"Setting up LoRA with rank={rank}, alpha={alpha}, dropout={dropout}")
        logger.info(f"Target modules: {target_modules}")

        # Create LoRA config
        self.peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, self.peft_config)

        # Log trainable parameters
        trainable_params, total_params = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        return self.model

    def setup_training_args(self) -> TrainingArguments:

        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine report destination
        report_to = ["wandb"] if self.wandb_enabled else []

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            gradient_checkpointing=self.gradient_checkpointing,
            fp16=self.fp16,
            bf16=self.bf16,
            optim=self.optim,
            logging_dir=str(output_path / "logs"),
            report_to=report_to,
            seed=self.seed,
            dataloader_num_workers=(
                self.config.get("dataloader_num_workers", 4) if self.device_type == "cuda" else 0
            ),
            dataloader_pin_memory=(self.device_type == "cuda"),
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        logger.info(f"Training arguments created. Output dir: {output_path}")
        return training_args

    def init_wandb(self, run_name: Optional[str] = None) -> None:
        """Initialize Weights & Biases logging.

        Args:
            run_name: Optional name for the W&B run.
        """
        if not self.wandb_enabled:
            logger.info("W&B logging disabled")
            return

        wandb_config = {
            "model_name": self.model_name,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }

        wandb.init(
            project=self.wandb_project,
            name=run_name or f"financial-llm-{self.lora_rank}r-{self.num_epochs}e",
            config=wandb_config,
        )

        logger.info(f"W&B initialized. Project: {self.wandb_project}")

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Setting up training...")

        # Initialize W&B
        self.init_wandb()

        # Create training arguments
        training_args = self.setup_training_args()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )

        logger.info("Starting training...")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Val dataset size: {len(val_dataset)}")

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Log final metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        logger.info("Training complete!")
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")

        # Close W&B
        if self.wandb_enabled:
            wandb.finish()

        return metrics

    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:

        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate(eval_dataset)

        self.trainer.log_metrics("eval", metrics)
        logger.info(f"Evaluation loss: {metrics.get('eval_loss', 'N/A')}")

        return metrics

    def save_model(self, output_path: Optional[str] = None) -> str:

        if self.model is None:
            raise ValueError("Model not loaded.")

        output_path = output_path or os.path.join(self.output_dir, "final_model")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_path}")

        # Save LoRA adapter
        self.model.save_pretrained(str(output_path))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))

        # Save training config
        import json
        config_path = output_path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved successfully to {output_path}")
        return str(output_path)

    def load_trained_model(self, adapter_path: str) -> PreTrainedModel:

        logger.info(f"Loading trained adapter from {adapter_path}")

        # Load base model if not already loaded
        if self.model is None:
            self.load_model()

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )

        logger.info("Adapter loaded successfully")
        return self.model

    def merge_and_save(self, output_path: str) -> str:

        if self.model is None:
            raise ValueError("Model not loaded.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Merging LoRA adapter and saving to {output_path}")

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save merged model
        merged_model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        logger.info("Merged model saved successfully")
        return str(output_path)

    def get_training_stats(self) -> Dict[str, Any]:

        stats = {
            "model_name": self.model_name,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.per_device_train_batch_size,
            "output_dir": self.output_dir,
        }

        if self.model is not None:
            trainable, total = self.model.get_nb_trainable_parameters()
            stats["trainable_params"] = trainable
            stats["total_params"] = total
            stats["trainable_percentage"] = 100 * trainable / total

        if self.trainer is not None and hasattr(self.trainer, "state"):
            stats["global_step"] = self.trainer.state.global_step
            stats["epochs_completed"] = self.trainer.state.epoch

        return stats


def create_trainer(config: Optional[Dict[str, Any]] = None) -> FinancialLLMTrainer:

    return FinancialLLMTrainer(config)
