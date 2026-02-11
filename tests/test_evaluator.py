"""Tests for evaluation metrics."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import FinancialLLMEvaluator, create_evaluator


class TestEvaluatorInit:
    """Test FinancialLLMEvaluator initialization."""

    def test_initialization(self):
        """Test evaluator initialization with mock model."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        assert evaluator.model is mock_model
        assert evaluator.tokenizer is mock_tokenizer
        mock_model.eval.assert_called_once()

    def test_custom_generation_config(self):
        """Test evaluator with custom generation config."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        gen_config = {
            "max_new_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.9,
        }

        evaluator = FinancialLLMEvaluator(
            mock_model, mock_tokenizer,
            generation_config=gen_config
        )

        assert evaluator.generation_config["max_new_tokens"] == 128
        assert evaluator.generation_config["temperature"] == 0.5

    def test_factory_function(self):
        """Test factory function creates evaluator."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = create_evaluator(mock_model, mock_tokenizer)

        assert isinstance(evaluator, FinancialLLMEvaluator)


class TestTextNormalization:
    """Test text normalization for comparison."""

    def test_normalize_text(self):
        """Test text normalization."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        # Test lowercasing
        assert evaluator._normalize_text("HELLO WORLD") == "hello world"

        # Test punctuation removal
        assert evaluator._normalize_text("Hello, World!") == "hello world"

        # Test whitespace normalization
        assert evaluator._normalize_text("Hello   World") == "hello world"

    def test_normalize_empty_text(self):
        """Test normalization of empty text."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        assert evaluator._normalize_text("") == ""


class TestExactMatch:
    """Test exact match computation."""

    def test_exact_match_identical(self):
        """Test exact match with identical texts."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        score = evaluator._compute_exact_match("Hello World", "hello world")
        assert score == 1.0

    def test_exact_match_different(self):
        """Test exact match with different texts."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        score = evaluator._compute_exact_match("Hello", "Goodbye")
        assert score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        score = evaluator._compute_exact_match("APPLE STOCK", "apple stock")
        assert score == 1.0


class TestTokenF1:
    """Test token-level F1 score computation."""

    def test_f1_perfect_match(self):
        """Test F1 with perfect match."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        result = evaluator._compute_token_f1("hello world", "hello world")

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_f1_partial_match(self):
        """Test F1 with partial match."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        result = evaluator._compute_token_f1("hello world", "hello there")

        assert 0 < result["f1"] < 1.0
        assert result["precision"] > 0
        assert result["recall"] > 0

    def test_f1_no_match(self):
        """Test F1 with no overlap."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        result = evaluator._compute_token_f1("apple stock", "orange juice")

        assert result["f1"] == 0.0

    def test_f1_empty_strings(self):
        """Test F1 with empty strings."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        result = evaluator._compute_token_f1("", "hello")

        assert result["f1"] == 0.0


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_estimate_cost_default(self):
        """Test cost estimation with default values."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        result = evaluator.estimate_cost(num_tokens=1_000_000)

        assert "cost_per_million_tokens" in result
        assert "self_hosted_cost" in result
        assert "gpt4_equivalent_cost" in result
        assert result["cost_per_million_tokens"] == 0.18

    def test_estimate_cost_savings(self):
        """Test savings calculations."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        result = evaluator.estimate_cost()

        # Should show savings compared to GPT-4
        assert result["savings_vs_gpt4"] > 0
        assert result["savings_percentage_vs_gpt4"] > 0


class TestLatencyEvaluation:
    """Test latency evaluation."""

    def test_latency_metrics_structure(self):
        """Test latency results have correct structure."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        # Mock the _generate_response method
        evaluator._generate_response = Mock(return_value=("Response", 100.0))

        result = evaluator.evaluate_latency(num_samples=5, warmup_samples=1)

        assert "latency_p50" in result
        assert "latency_p95" in result
        assert "latency_p99" in result
        assert "latency_mean" in result

    def test_latency_empty_results(self):
        """Test latency with failing generation."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        # Mock the _generate_response to raise exceptions
        evaluator._generate_response = Mock(side_effect=Exception("Error"))

        result = evaluator.evaluate_latency(num_samples=3, warmup_samples=0)

        assert result["latency_p50"] == 0.0
        assert result["latency_mean"] == 0.0


class TestEvaluateAll:
    """Test comprehensive evaluation."""

    def test_evaluate_all_structure(self):
        """Test evaluate_all returns all expected metrics."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        # Mock all evaluation methods
        evaluator.evaluate_reasoning = Mock(return_value={"reasoning_accuracy": 0.8})
        evaluator.evaluate_qa = Mock(return_value={"qa_f1": 0.75})
        evaluator.evaluate_ner = Mock(return_value={"ner_f1": 0.82})
        evaluator.evaluate_latency = Mock(return_value={"latency_p99": 150.0})
        evaluator.estimate_cost = Mock(return_value={"cost_per_million_tokens": 0.18})

        # Create mock dataset
        mock_dataset = Mock()

        result = evaluator.evaluate_all(mock_dataset, skip_latency=False)

        assert "reasoning_accuracy" in result
        assert "qa_f1" in result
        assert "ner_f1" in result
        assert "latency_p99" in result
        assert "cost_per_million_tokens" in result

    def test_evaluate_all_skip_latency(self):
        """Test evaluate_all can skip latency."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_tokenizer = Mock()

        evaluator = FinancialLLMEvaluator(mock_model, mock_tokenizer)

        # Mock all evaluation methods
        evaluator.evaluate_reasoning = Mock(return_value={"reasoning_accuracy": 0.8})
        evaluator.evaluate_qa = Mock(return_value={"qa_f1": 0.75})
        evaluator.evaluate_ner = Mock(return_value={"ner_f1": 0.82})
        evaluator.evaluate_latency = Mock(return_value={"latency_p99": 150.0})
        evaluator.estimate_cost = Mock(return_value={"cost_per_million_tokens": 0.18})

        mock_dataset = Mock()

        result = evaluator.evaluate_all(mock_dataset, skip_latency=True)

        # Latency should not be called
        evaluator.evaluate_latency.assert_not_called()
