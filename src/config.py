
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration settings."""

    name: str = Field(
        default="microsoft/phi-3.5-mini-instruct",
        description="HuggingFace model name or path"
    )
    model_id: str = Field(default="phi-3.5-mini", description="Short model identifier")
    model_type: str = Field(default="causal-lm", description="Model type")
    num_parameters: int = Field(default=3_840_000_000, description="Number of parameters")
    trust_remote_code: bool = Field(default=True, description="Trust remote code from HuggingFace")


class QuantizationConfig(BaseModel):
    """Quantization settings for model loading."""

    type: str = Field(default="8bit", description="Quantization type: 8bit, 4bit, or none")
    load_in_8bit: bool = Field(default=True, description="Load model in 8-bit precision")
    load_in_4bit: bool = Field(default=False, description="Load model in 4-bit precision")

    @field_validator("type")
    @classmethod
    def validate_quantization_type(cls, v: str) -> str:
        """Validate quantization type."""
        valid_types = ["8bit", "4bit", "none"]
        if v not in valid_types:
            raise ValueError(f"Quantization type must be one of {valid_types}")
        return v


class InferenceConfig(BaseModel):
    """Inference settings."""

    device: str = Field(default="cuda", description="Device to run inference on")
    precision: str = Field(default="float16", description="Precision for inference")
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)


class GenerationConfig(BaseModel):
    """Text generation settings."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="Repetition penalty")
    max_new_tokens: int = Field(default=512, ge=1, description="Maximum new tokens to generate")
    min_new_tokens: int = Field(default=10, ge=0, description="Minimum new tokens to generate")
    length_penalty: float = Field(default=1.0, description="Length penalty for beam search")
    do_sample: bool = Field(default=True, description="Use sampling for generation")


class TokenizerConfig(BaseModel):
    """Tokenizer settings."""

    max_length: int = Field(default=512, ge=1, description="Maximum sequence length")
    padding: str = Field(default="max_length", description="Padding strategy")
    truncation: bool = Field(default=True, description="Whether to truncate sequences")


class DatasetConfig(BaseModel):
    """Dataset configuration settings."""

    name: str = Field(
        default="Josephgflowers/Finance-Instruct-500k",
        description="HuggingFace dataset name"
    )
    subset_size: int = Field(default=50000, ge=1, description="Number of examples to use")
    max_seq_length: int = Field(default=512, ge=1, description="Maximum sequence length")

    # Split ratios
    train_ratio: float = Field(default=0.8, ge=0.0, le=1.0, description="Training split ratio")
    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Validation split ratio")
    test_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Test split ratio")

    # Filtering
    min_length: int = Field(default=50, ge=0, description="Minimum token length")
    max_length: int = Field(default=1000, ge=1, description="Maximum token length")

    # Processing
    num_proc: int = Field(default=4, ge=1, description="Number of processes for data loading")
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    seed: int = Field(default=42, description="Random seed for shuffling")

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        """Validate that ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Ratio must be between 0 and 1")
        return v


class LoraConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int = Field(default=16, ge=1, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(
        default=["qkv_proj", "o_proj"],
        description="Modules to apply LoRA to"
    )
    bias: str = Field(default="none", description="Bias training strategy")
    task_type: str = Field(default="CAUSAL_LM", description="Task type for LoRA")


class WandbConfig(BaseModel):
    """Weights & Biases logging configuration."""

    project: str = Field(default="financial-llm-advisor", description="W&B project name")
    entity: Optional[str] = Field(default=None, description="W&B entity/team name")
    enabled: bool = Field(default=True, description="Enable W&B logging")
    log_model: bool = Field(default=True, description="Log model artifacts")
    run_name: Optional[str] = Field(default=None, description="Run name")
    tags: List[str] = Field(default_factory=list, description="Tags for the run")


class TrainingConfig(BaseModel):
    """Training configuration settings."""

    # Training hyperparameters
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=16, ge=1, description="Training batch size")
    per_device_eval_batch_size: int = Field(default=16, ge=1, description="Evaluation batch size")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    warmup_steps: int = Field(default=500, ge=0, description="Warmup steps")
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Warmup ratio")
    max_steps: int = Field(default=-1, description="Maximum training steps (-1 for epochs)")

    # Logging and saving
    logging_steps: int = Field(default=100, ge=1, description="Logging frequency")
    eval_steps: int = Field(default=500, ge=1, description="Evaluation frequency")
    save_steps: int = Field(default=500, ge=1, description="Checkpoint save frequency")
    save_total_limit: int = Field(default=3, ge=1, description="Maximum checkpoints to keep")

    # Training optimizations
    gradient_checkpointing: bool = Field(default=True, description="Use gradient checkpointing")
    fp16: bool = Field(default=False, description="Use FP16 mixed precision")
    bf16: bool = Field(default=False, description="Use BF16 mixed precision")
    optim: str = Field(default="adamw_torch", description="Optimizer")

    # Directories
    output_dir: str = Field(default="./checkpoints", description="Output directory")
    logging_dir: str = Field(default="./logs", description="Logging directory")

    # Strategies
    eval_strategy: str = Field(default="steps", description="Evaluation strategy")
    save_strategy: str = Field(default="steps", description="Save strategy")

    # Other
    seed: int = Field(default=42, description="Random seed")
    dataloader_num_workers: int = Field(default=4, ge=0, description="Dataloader workers")
    remove_unused_columns: bool = Field(default=False, description="Remove unused columns")
    label_names: List[str] = Field(default=["labels"], description="Label column names")

    # LoRA config
    lora: LoraConfig = Field(default_factory=LoraConfig)

    # W&B config
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class Config:

    def __init__(self, config_path: Optional[Union[str, Path]] = None):

        # Initialize with defaults
        self.model = ModelConfig()
        self.dataset = DatasetConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.generation = GenerationConfig()
        self.tokenizer = TokenizerConfig()
        self.system_prompt: str = (
            "You are a financial advisor assistant with expertise in investment analysis, "
            "portfolio management, and financial markets. You provide clear, data-driven "
            "investment insights with proper reasoning and confidence levels."
        )

        if config_path:
            self.load_from_path(config_path)

    def load_from_path(self, config_path: Union[str, Path]) -> None:

        path = Path(config_path)

        if path.is_file():
            self.load_from_yaml(path)
        elif path.is_dir():
            # Load all YAML files in directory
            for yaml_file in path.glob("*.yaml"):
                self.load_from_yaml(yaml_file)
            for yml_file in path.glob("*.yml"):
                self.load_from_yaml(yml_file)
        else:
            raise FileNotFoundError(f"Config path not found: {config_path}")

    def load_from_yaml(self, yaml_path: Union[str, Path]) -> None:
        yaml_path = Path(yaml_path)
        logger.info(f"Loading configuration from {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            logger.warning(f"Empty config file: {yaml_path}")
            return

        self._update_from_dict(config_dict)

    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:

        # Update model config
        if "model" in config_dict:
            model_data = {**self.model.model_dump(), **config_dict["model"]}
            self.model = ModelConfig(**model_data)
        elif "model_name" in config_dict:
            # Handle flat model config format
            model_data = self.model.model_dump()
            if "model_name" in config_dict:
                model_data["name"] = config_dict["model_name"]
            if "model_id" in config_dict:
                model_data["model_id"] = config_dict["model_id"]
            if "model_type" in config_dict:
                model_data["model_type"] = config_dict["model_type"]
            if "num_parameters" in config_dict:
                model_data["num_parameters"] = config_dict["num_parameters"]
            self.model = ModelConfig(**model_data)

        # Update inference config
        if "inference" in config_dict:
            inference_data = config_dict["inference"]
            if "quantization" in inference_data:
                quant_config = QuantizationConfig(**inference_data["quantization"])
                inference_data = {**inference_data, "quantization": quant_config}
            self.inference = InferenceConfig(**inference_data)

        # Update generation config
        if "generation" in config_dict:
            gen_data = {**self.generation.model_dump(), **config_dict["generation"]}
            self.generation = GenerationConfig(**gen_data)

        # Update tokenizer config
        if "tokenizer" in config_dict:
            tok_data = {**self.tokenizer.model_dump(), **config_dict["tokenizer"]}
            self.tokenizer = TokenizerConfig(**tok_data)

        # Update dataset config
        if "dataset" in config_dict:
            dataset_data = config_dict["dataset"]
            # Handle nested split_ratio
            if "split_ratio" in dataset_data:
                split_ratio = dataset_data.pop("split_ratio")
                dataset_data["train_ratio"] = split_ratio.get("train", 0.8)
                dataset_data["val_ratio"] = split_ratio.get("validation", 0.1)
                dataset_data["test_ratio"] = split_ratio.get("test", 0.1)
            # Handle nested filtering
            if "filtering" in dataset_data:
                filtering = dataset_data.pop("filtering")
                dataset_data["min_length"] = filtering.get("min_length", 50)
                dataset_data["max_length"] = filtering.get("max_length", 1000)
            ds_data = {**self.dataset.model_dump(), **dataset_data}
            self.dataset = DatasetConfig(**ds_data)

        # Update training config
        if "training" in config_dict:
            training_data = config_dict["training"].copy()
            self._merge_training_config(training_data)

        # Handle top-level training params (for training_config.yaml format)
        training_keys = {
            "num_epochs", "per_device_train_batch_size", "per_device_eval_batch_size",
            "learning_rate", "warmup_steps", "gradient_accumulation_steps",
            "weight_decay", "logging_steps", "eval_steps", "save_steps",
            "save_total_limit", "gradient_checkpointing", "fp16", "bf16", "optim",
            "output_dir", "logging_dir", "eval_strategy", "save_strategy", "seed"
        }
        top_level_training = {k: v for k, v in config_dict.items() if k in training_keys}
        if top_level_training:
            self._merge_training_config(top_level_training)

        # Update LoRA config
        if "lora" in config_dict:
            lora_data = {**self.training.lora.model_dump(), **config_dict["lora"]}
            self.training.lora = LoraConfig(**lora_data)

        # Update W&B config
        if "wandb" in config_dict:
            wandb_data = {**self.training.wandb.model_dump(), **config_dict["wandb"]}
            self.training.wandb = WandbConfig(**wandb_data)

        # Update system prompt
        if "system_prompt" in config_dict:
            self.system_prompt = config_dict["system_prompt"]

    def _merge_training_config(self, training_data: Dict[str, Any]) -> None:

        # Extract nested configs
        lora_data = training_data.pop("lora", None)
        wandb_data = training_data.pop("wandb", None)

        # Merge main training config
        current_data = self.training.model_dump()
        # Remove nested configs from current data for merge
        current_data.pop("lora", None)
        current_data.pop("wandb", None)

        merged = {**current_data, **training_data}
        merged["lora"] = self.training.lora
        merged["wandb"] = self.training.wandb

        self.training = TrainingConfig(**merged)

        # Update nested configs if provided
        if lora_data:
            lora_merged = {**self.training.lora.model_dump(), **lora_data}
            self.training.lora = LoraConfig(**lora_merged)

        if wandb_data:
            wandb_merged = {**self.training.wandb.model_dump(), **wandb_data}
            self.training.wandb = WandbConfig(**wandb_merged)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "model": self.model.model_dump(),
            "dataset": self.dataset.model_dump(),
            "training": self.training.model_dump(),
            "inference": self.inference.model_dump(),
            "generation": self.generation.model_dump(),
            "tokenizer": self.tokenizer.model_dump(),
            "system_prompt": self.system_prompt,
        }

    def save_to_yaml(self, output_path: Union[str, Path]) -> None:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")

    def get_training_args_dict(self) -> Dict[str, Any]:
        """Get dictionary suitable for HuggingFace TrainingArguments.

        Returns:
            Dictionary of training arguments.
        """
        return {
            "output_dir": self.training.output_dir,
            "num_train_epochs": self.training.num_epochs,
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "per_device_eval_batch_size": self.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "warmup_steps": self.training.warmup_steps,
            "warmup_ratio": self.training.warmup_ratio,
            "max_steps": self.training.max_steps,
            "logging_steps": self.training.logging_steps,
            "eval_steps": self.training.eval_steps,
            "save_steps": self.training.save_steps,
            "save_total_limit": self.training.save_total_limit,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "fp16": self.training.fp16,
            "bf16": self.training.bf16,
            "optim": self.training.optim,
            "logging_dir": self.training.logging_dir,
            "evaluation_strategy": self.training.eval_strategy,
            "save_strategy": self.training.save_strategy,
            "seed": self.training.seed,
            "dataloader_num_workers": self.training.dataloader_num_workers,
            "remove_unused_columns": self.training.remove_unused_columns,
            "label_names": self.training.label_names,
        }

    def get_lora_config_dict(self) -> Dict[str, Any]:
        """Get dictionary suitable for PEFT LoraConfig.

        Returns:
            Dictionary of LoRA configuration.
        """
        return {
            "r": self.training.lora.rank,
            "lora_alpha": self.training.lora.alpha,
            "lora_dropout": self.training.lora.dropout,
            "target_modules": self.training.lora.target_modules,
            "bias": self.training.lora.bias,
            "task_type": self.training.lora.task_type,
        }

    def get_generation_config_dict(self) -> Dict[str, Any]:
        """Get dictionary suitable for model.generate().

        Returns:
            Dictionary of generation configuration.
        """
        return self.generation.model_dump()

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"Config(model={self.model.name}, "
            f"dataset={self.dataset.name}, "
            f"epochs={self.training.num_epochs}, "
            f"lora_rank={self.training.lora.rank})"
        )


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:

    return Config(config_path)
