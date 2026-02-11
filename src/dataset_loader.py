from typing import Tuple, Optional, Dict, Any, List, Union, Callable
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import DataLoader
import torch
import logging

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "Josephgflowers/Finance-Instruct-500k"
DEFAULT_MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
DEFAULT_SUBSET_SIZE = 50000
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_MIN_LENGTH = 50
DEFAULT_MAX_LENGTH = 1000
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42


class DatasetLoader:

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # Extract configuration with defaults
        self.dataset_name = self.config.get("name", DEFAULT_DATASET_NAME)
        self.subset_size = self.config.get("subset_size", DEFAULT_SUBSET_SIZE)
        self.max_seq_length = self.config.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        self.min_length = self.config.get("min_length", DEFAULT_MIN_LENGTH)
        self.max_length = self.config.get("max_length", DEFAULT_MAX_LENGTH)
        self.train_ratio = self.config.get("train_ratio", DEFAULT_TRAIN_RATIO)
        self.val_ratio = self.config.get("val_ratio", DEFAULT_VAL_RATIO)
        self.test_ratio = self.config.get("test_ratio", DEFAULT_TEST_RATIO)
        self.num_proc = self.config.get("num_proc", 4)
        self.shuffle = self.config.get("shuffle", True)
        self.seed = self.config.get("seed", DEFAULT_SEED)

        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.raw_dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def load_tokenizer(
        self,
        model_name: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> PreTrainedTokenizer:

        model_name = model_name or DEFAULT_MODEL_NAME
        logger.info(f"Loading tokenizer from {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")
        return self.tokenizer

    def load_raw_dataset(self, download: bool = True) -> Dataset:

        logger.info(f"Loading dataset: {self.dataset_name}")
        logger.info(f"Subset size: {self.subset_size}")

        try:
            # Load the dataset
            dataset = load_dataset(
                self.dataset_name,
                split="train",
                download_mode="reuse_cache_if_exists" if not download else None,
            )

            logger.info(f"Raw dataset loaded with {len(dataset)} examples")

            # Shuffle and select subset
            if self.shuffle:
                dataset = dataset.shuffle(seed=self.seed)

            if self.subset_size and self.subset_size < len(dataset):
                dataset = dataset.select(range(self.subset_size))
                logger.info(f"Selected {self.subset_size} examples")

            self.raw_dataset = dataset
            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise ValueError(f"Failed to load dataset {self.dataset_name}: {e}")

    def _get_text_length(self, example: Dict[str, Any]) -> int:

        text_parts = []

        # Handle different dataset formats
        if "user" in example:
            text_parts.append(str(example.get("user", "")))
        if "assistant" in example:
            text_parts.append(str(example.get("assistant", "")))
        if "instruction" in example:
            text_parts.append(str(example.get("instruction", "")))
        if "input" in example:
            text_parts.append(str(example.get("input", "")))
        if "output" in example:
            text_parts.append(str(example.get("output", "")))
        if "text" in example:
            text_parts.append(str(example.get("text", "")))
        if "question" in example:
            text_parts.append(str(example.get("question", "")))
        if "answer" in example:
            text_parts.append(str(example.get("answer", "")))

        combined_text = " ".join(text_parts)

        # Rough token estimate (words / 0.75)
        word_count = len(combined_text.split())
        token_estimate = int(word_count / 0.75)

        return token_estimate

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter dataset by length and quality.

        Removes examples that are too short or too long based on
        estimated token counts.

        Args:
            dataset: Raw dataset to filter.

        Returns:
            Filtered dataset.
        """
        initial_size = len(dataset)
        logger.info(f"Filtering dataset. Initial size: {initial_size}")
        logger.info(f"Filter criteria: {self.min_length} < tokens < {self.max_length}")

        def filter_by_length(example: Dict[str, Any]) -> bool:
            """Filter function for length-based filtering."""
            length = self._get_text_length(example)
            return self.min_length <= length <= self.max_length

        def filter_valid_content(example: Dict[str, Any]) -> bool:
            """Filter function for content validation."""
            # Finance-Instruct-500k format
            has_user = bool(str(example.get("user", "")).strip())
            has_assistant = bool(str(example.get("assistant", "")).strip())

            # Instruction/output format
            has_instruction = bool(str(example.get("instruction", "")).strip())
            has_output = bool(str(example.get("output", "")).strip())

            # Q&A format
            has_question = bool(str(example.get("question", "")).strip())
            has_answer = bool(str(example.get("answer", "")).strip())

            # Plain text format
            has_text = bool(str(example.get("text", "")).strip())

            return (
                (has_user and has_assistant)
                or (has_instruction and has_output)
                or (has_question and has_answer)
                or has_text
            )

        # Apply filters
        filtered_dataset = dataset.filter(
            filter_valid_content,
            num_proc=self.num_proc,
            desc="Filtering invalid content",
        )

        filtered_dataset = filtered_dataset.filter(
            filter_by_length,
            num_proc=self.num_proc,
            desc="Filtering by length",
        )

        final_size = len(filtered_dataset)
        removed = initial_size - final_size
        logger.info(f"Filtered dataset size: {final_size} (removed {removed})")

        return filtered_dataset

    def _format_example(self, example: Dict[str, Any]) -> str:

        # System prompt for financial advisor
        system_prompt = (
            "You are a financial advisor assistant with expertise in investment analysis, "
            "portfolio management, and financial markets. You provide clear, data-driven "
            "investment insights with proper reasoning and confidence levels."
        )

        # Handle different dataset formats
        if "user" in example and "assistant" in example:
            user_content = str(example.get("user", "")).strip()
            output = str(example.get("assistant", "")).strip()
            # Use dataset's system prompt if provided, otherwise use default
            ds_system = str(example.get("system", "")).strip()
            if ds_system:
                system_prompt = ds_system

        elif "instruction" in example and "output" in example:
            instruction = str(example.get("instruction", "")).strip()
            input_text = str(example.get("input", "")).strip()
            output = str(example.get("output", "")).strip()

            if input_text:
                user_content = f"{instruction}\n\nContext: {input_text}"
            else:
                user_content = instruction

        elif "question" in example and "answer" in example:
            user_content = str(example.get("question", "")).strip()
            output = str(example.get("answer", "")).strip()

        elif "text" in example:
            # Simple text format - try to split into prompt/response
            text = str(example.get("text", "")).strip()
            if "###" in text:
                parts = text.split("###")
                user_content = parts[0].strip()
                output = "###".join(parts[1:]).strip()
            else:
                # Use the whole text as output with a generic prompt
                user_content = "Provide financial analysis and insights."
                output = text

        else:
            # Fallback
            user_content = str(example.get("prompt", example.get("input", ""))).strip()
            output = str(example.get("response", example.get("output", ""))).strip()

        # Format in Phi-3.5 chat format
        formatted = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_content}<|end|>\n"
            f"<|assistant|>\n{output}<|end|>"
        )

        return formatted

    def _tokenize_function(
        self,
        examples: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:

        # Format each example
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            example = {key: examples[key][i] for key in examples.keys()}
            formatted_text = self._format_example(example)
            texts.append(formatted_text)

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )

        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        # Mask padding tokens in labels
        for i, label_ids in enumerate(tokenized["labels"]):
            tokenized["labels"][i] = [
                -100 if token_id == self.tokenizer.pad_token_id else token_id
                for token_id in label_ids
            ]

        return tokenized

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Dataset:

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if self.tokenizer is None:
            self.load_tokenizer()

        logger.info("Tokenizing dataset...")

        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        logger.info(f"Tokenization complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset

    def create_splits(
        self,
        dataset: Dataset,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train/validation/test splits.
        """
        logger.info("Creating train/val/test splits...")
        logger.info(
            f"Split ratios: train={self.train_ratio}, "
            f"val={self.val_ratio}, test={self.test_ratio}"
        )

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * self.train_ratio)
        val_size = int(total_size * self.val_ratio)
        # Test gets the remainder

        # Shuffle and split
        shuffled = dataset.shuffle(seed=self.seed)

        train_dataset = shuffled.select(range(train_size))
        val_dataset = shuffled.select(range(train_size, train_size + val_size))
        test_dataset = shuffled.select(range(train_size + val_size, total_size))

        logger.info(
            f"Split sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        return train_dataset, val_dataset, test_dataset

    def load_and_process_dataset(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Tuple[Dataset, Dataset]:
        logger.info("Starting data loading and processing pipeline...")

        # Load tokenizer if needed
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.tokenizer is None:
            self.load_tokenizer()

        # Load raw dataset
        raw_dataset = self.load_raw_dataset()

        # Filter dataset
        filtered_dataset = self.filter_dataset(raw_dataset)

        # Create splits (before tokenization for efficiency)
        train_ds, val_ds, test_ds = self.create_splits(filtered_dataset)

        # Tokenize each split
        self.train_dataset = self.tokenize_dataset(train_ds)
        self.val_dataset = self.tokenize_dataset(val_ds)
        self.test_dataset = self.tokenize_dataset(test_ds)

        logger.info("Data processing pipeline complete!")
        return self.train_dataset, self.val_dataset

    def load_test_dataset(self) -> Dataset:
        if self.test_dataset is None:
            logger.info("Test dataset not loaded. Running full pipeline...")
            self.load_and_process_dataset()

        return self.test_dataset

    def get_dataloaders(
        self,
        train_batch_size: int = 16,
        eval_batch_size: int = 16,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_and_process_dataset first.")

        # Set format for PyTorch
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        logger.info(
            f"DataLoaders created: train={len(train_dataloader)} batches, "
            f"val={len(val_dataloader)} batches, test={len(test_dataloader)} batches"
        )

        return train_dataloader, val_dataloader, test_dataloader

    def get_sample_batch(self, split: str = "train", batch_size: int = 4) -> Dict[str, torch.Tensor]:

        dataset_map = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

        dataset = dataset_map.get(split)
        if dataset is None:
            raise ValueError(f"Dataset split '{split}' not loaded.")

        # Select a random batch
        indices = list(range(min(batch_size, len(dataset))))
        batch = dataset.select(indices)
        batch.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return {
            "input_ids": torch.stack([batch[i]["input_ids"] for i in range(len(batch))]),
            "attention_mask": torch.stack([batch[i]["attention_mask"] for i in range(len(batch))]),
            "labels": torch.stack([batch[i]["labels"] for i in range(len(batch))]),
        }

    def decode_example(self, input_ids: Union[List[int], torch.Tensor]) -> str:

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        return self.tokenizer.decode(input_ids, skip_special_tokens=False)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            "dataset_name": self.dataset_name,
            "subset_size": self.subset_size,
            "max_seq_length": self.max_seq_length,
            "filter_min_length": self.min_length,
            "filter_max_length": self.max_length,
        }

        if self.train_dataset is not None:
            stats["train_size"] = len(self.train_dataset)
        if self.val_dataset is not None:
            stats["val_size"] = len(self.val_dataset)
        if self.test_dataset is not None:
            stats["test_size"] = len(self.test_dataset)

        if self.tokenizer is not None:
            stats["vocab_size"] = self.tokenizer.vocab_size

        return stats


def create_dataset_loader(config: Optional[Dict[str, Any]] = None) -> DatasetLoader:
    return DatasetLoader(config)
