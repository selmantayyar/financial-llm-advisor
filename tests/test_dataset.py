"""Tests for dataset loading and processing."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import DatasetLoader, create_dataset_loader


class TestDatasetLoaderInit:
    """Test DatasetLoader initialization."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        loader = DatasetLoader()

        assert loader.dataset_name == "Josephgflowers/Finance-Instruct-500k"
        assert loader.subset_size == 50000
        assert loader.max_seq_length == 512
        assert loader.min_length == 50
        assert loader.max_length == 1000
        assert loader.train_ratio == 0.8
        assert loader.val_ratio == 0.1
        assert loader.test_ratio == 0.1

    def test_custom_configuration(self):
        """Test initialization with custom config."""
        config = {
            "name": "custom/dataset",
            "subset_size": 1000,
            "max_seq_length": 256,
            "min_length": 10,
            "max_length": 500,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        }
        loader = DatasetLoader(config)

        assert loader.dataset_name == "custom/dataset"
        assert loader.subset_size == 1000
        assert loader.max_seq_length == 256
        assert loader.min_length == 10
        assert loader.max_length == 500
        assert loader.train_ratio == 0.7
        assert loader.val_ratio == 0.15
        assert loader.test_ratio == 0.15

    def test_factory_function(self):
        """Test factory function creates DatasetLoader."""
        loader = create_dataset_loader({"subset_size": 500})

        assert isinstance(loader, DatasetLoader)
        assert loader.subset_size == 500


class TestTextLength:
    """Test text length calculation."""

    def test_instruction_output_format(self):
        """Test length calculation for instruction/output format."""
        loader = DatasetLoader()

        example = {
            "instruction": "Analyze the stock market",
            "input": "Context about Apple stock",
            "output": "Based on my analysis..."
        }

        length = loader._get_text_length(example)
        assert length > 0
        assert isinstance(length, int)

    def test_question_answer_format(self):
        """Test length calculation for question/answer format."""
        loader = DatasetLoader()

        example = {
            "question": "What is the P/E ratio?",
            "answer": "The P/E ratio is..."
        }

        length = loader._get_text_length(example)
        assert length > 0

    def test_empty_example(self):
        """Test length calculation for empty example."""
        loader = DatasetLoader()

        example = {}
        length = loader._get_text_length(example)
        assert length == 0


class TestFormatExample:
    """Test example formatting."""

    def test_instruction_format(self):
        """Test formatting instruction/input/output."""
        loader = DatasetLoader()

        example = {
            "instruction": "Analyze Apple stock",
            "input": "Current price: $150",
            "output": "Based on the analysis..."
        }

        formatted = loader._format_example(example)

        assert "<|system|>" in formatted
        assert "<|user|>" in formatted
        assert "<|assistant|>" in formatted
        assert "Analyze Apple stock" in formatted
        assert "Current price: $150" in formatted
        assert "Based on the analysis..." in formatted

    def test_question_answer_format(self):
        """Test formatting question/answer."""
        loader = DatasetLoader()

        example = {
            "question": "What is EBITDA?",
            "answer": "EBITDA stands for..."
        }

        formatted = loader._format_example(example)

        assert "<|system|>" in formatted
        assert "<|user|>" in formatted
        assert "What is EBITDA?" in formatted
        assert "EBITDA stands for..." in formatted

    def test_text_format(self):
        """Test formatting text field."""
        loader = DatasetLoader()

        example = {
            "text": "Some financial text content"
        }

        formatted = loader._format_example(example)

        assert "<|system|>" in formatted
        assert "Some financial text content" in formatted


class TestFilterDataset:
    """Test dataset filtering."""

    def test_filter_by_length(self):
        """Test filtering removes examples outside length range."""
        loader = DatasetLoader({"min_length": 10, "max_length": 100})

        # Create mock dataset
        examples = [
            {"instruction": "Short", "output": "S"},  # Too short
            {"instruction": "This is a medium length instruction that should pass the filter",
             "output": "This is a valid output with enough content."},  # Valid
            {"instruction": "x" * 2000, "output": "x" * 2000},  # Too long
        ]
        dataset = Dataset.from_list(examples)

        filtered = loader.filter_dataset(dataset)

        # Should have fewer examples after filtering
        assert len(filtered) < len(dataset)

    def test_filter_invalid_content(self):
        """Test filtering removes examples with missing fields."""
        loader = DatasetLoader()

        examples = [
            {"instruction": "Valid instruction", "output": "Valid output"},
            {"instruction": "", "output": "Output without instruction"},
            {"instruction": "Instruction without output", "output": ""},
        ]
        dataset = Dataset.from_list(examples)

        filtered = loader.filter_dataset(dataset)

        # Should filter out invalid examples
        assert len(filtered) <= len(dataset)


class TestCreateSplits:
    """Test train/val/test split creation."""

    def test_split_ratios(self):
        """Test splits have correct ratios."""
        loader = DatasetLoader({
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1
        })

        # Create mock dataset
        examples = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)]
        dataset = Dataset.from_list(examples)

        train, val, test = loader.create_splits(dataset)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_splits_are_disjoint(self):
        """Test splits don't overlap."""
        loader = DatasetLoader()

        examples = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)]
        dataset = Dataset.from_list(examples)

        train, val, test = loader.create_splits(dataset)

        # Total should equal original
        assert len(train) + len(val) + len(test) == len(dataset)


class TestGetDatasetStats:
    """Test dataset statistics."""

    def test_stats_before_loading(self):
        """Test stats when no data is loaded."""
        loader = DatasetLoader()

        stats = loader.get_dataset_stats()

        assert "dataset_name" in stats
        assert "subset_size" in stats
        assert stats["dataset_name"] == "Josephgflowers/Finance-Instruct-500k"

    def test_stats_after_loading(self):
        """Test stats include split sizes after loading."""
        loader = DatasetLoader()

        # Manually set datasets
        examples = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(100)]
        dataset = Dataset.from_list(examples)

        loader.train_dataset = dataset
        loader.val_dataset = dataset
        loader.test_dataset = dataset

        stats = loader.get_dataset_stats()

        assert stats["train_size"] == 100
        assert stats["val_size"] == 100
        assert stats["test_size"] == 100


class TestTokenization:
    """Test tokenization functionality."""

    @patch("src.dataset_loader.AutoTokenizer")
    def test_load_tokenizer(self, mock_tokenizer_class):
        """Test tokenizer loading."""
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 32000
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        loader = DatasetLoader()
        tokenizer = loader.load_tokenizer()

        assert tokenizer is not None
        mock_tokenizer_class.from_pretrained.assert_called_once()

    @patch("src.dataset_loader.AutoTokenizer")
    def test_tokenize_function(self, mock_tokenizer_class):
        """Test tokenization function."""
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 32000
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3, 0, 0]],
            "attention_mask": [[1, 1, 1, 0, 0]]
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        loader = DatasetLoader()
        loader.tokenizer = mock_tokenizer

        examples = {
            "instruction": ["What is EBITDA?"],
            "output": ["EBITDA is..."]
        }

        result = loader._tokenize_function(examples)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
