"""Tests for inference functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import (
    FinancialAdvisor,
    AnalysisRequest,
    AnalysisResponse,
    BatchRequest,
    HealthResponse,
    create_advisor,
)


class TestAnalysisRequest:
    """Test AnalysisRequest model."""

    def test_default_values(self):
        """Test request has correct default values."""
        request = AnalysisRequest(query="What is EBITDA?")

        assert request.query == "What is EBITDA?"
        assert request.context is None
        assert request.max_tokens == 512
        assert request.temperature == 0.7
        assert request.top_p == 0.95

    def test_custom_values(self):
        """Test request with custom values."""
        request = AnalysisRequest(
            query="Analyze Apple stock",
            context="Apple reported Q4 earnings...",
            max_tokens=256,
            temperature=0.5,
            top_p=0.9,
        )

        assert request.query == "Analyze Apple stock"
        assert request.context == "Apple reported Q4 earnings..."
        assert request.max_tokens == 256
        assert request.temperature == 0.5
        assert request.top_p == 0.9


class TestAnalysisResponse:
    """Test AnalysisResponse model."""

    def test_response_structure(self):
        """Test response has correct structure."""
        response = AnalysisResponse(
            analysis="Apple stock shows positive momentum...",
            confidence=0.85,
            entities=["Apple", "AAPL"],
            latency_ms=150.5,
        )

        assert response.analysis == "Apple stock shows positive momentum..."
        assert response.confidence == 0.85
        assert response.entities == ["Apple", "AAPL"]
        assert response.latency_ms == 150.5


class TestBatchRequest:
    """Test BatchRequest model."""

    def test_batch_request(self):
        """Test batch request structure."""
        request = BatchRequest(
            queries=["Query 1", "Query 2", "Query 3"],
            max_tokens=256,
        )

        assert len(request.queries) == 3
        assert request.max_tokens == 256


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_health_response(self):
        """Test health response structure."""
        response = HealthResponse(
            status="ok",
            model_loaded=True,
            device="cuda",
        )

        assert response.status == "ok"
        assert response.model_loaded is True
        assert response.device == "cuda"


class TestFinancialAdvisorInit:
    """Test FinancialAdvisor initialization."""

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test advisor initializes correctly."""
        # Setup mocks
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        advisor = FinancialAdvisor(quantization="none")

        assert advisor.model is not None
        assert advisor.tokenizer is not None
        mock_model.eval.assert_called_once()

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_custom_system_prompt(self, mock_tokenizer_class, mock_model_class):
        """Test advisor with custom system prompt."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        custom_prompt = "You are a custom financial advisor."
        advisor = FinancialAdvisor(
            quantization="none",
            system_prompt=custom_prompt,
        )

        assert advisor.system_prompt == custom_prompt


class TestConfidenceComputation:
    """Test confidence score computation."""

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_confidence_with_scores(self, mock_tokenizer_class, mock_model_class):
        """Test confidence computation with generation scores."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        advisor = FinancialAdvisor(quantization="none")

        # Create mock output_ids and scores
        output_ids = torch.tensor([1, 2, 3, 4, 5])

        # Create mock scores (logits for each position)
        scores = []
        for i in range(4):
            logits = torch.randn(1, 100)
            # Make the correct token have high probability
            logits[0, output_ids[i + 1].item()] = 5.0
            scores.append(logits)

        confidence = advisor._compute_confidence(output_ids, scores)

        assert 0.0 <= confidence <= 1.0

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_confidence_empty_scores(self, mock_tokenizer_class, mock_model_class):
        """Test confidence with empty scores."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        advisor = FinancialAdvisor(quantization="none")

        output_ids = torch.tensor([1, 2, 3])
        confidence = advisor._compute_confidence(output_ids, [])

        assert confidence == 0.5  # Default confidence


class TestBatchAnalyze:
    """Test batch analysis functionality."""

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_batch_analyze(self, mock_tokenizer_class, mock_model_class):
        """Test batch analysis returns correct number of results."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        advisor = FinancialAdvisor(quantization="none")

        # Mock the analyze method
        advisor.analyze = Mock(return_value={
            "analysis": "Test analysis",
            "confidence": 0.8,
            "entities": [],
            "latency_ms": 100.0,
        })

        queries = ["Query 1", "Query 2", "Query 3"]
        results = advisor.batch_analyze(queries)

        assert len(results) == 3
        assert advisor.analyze.call_count == 3

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_batch_analyze_handles_errors(self, mock_tokenizer_class, mock_model_class):
        """Test batch analysis handles individual errors gracefully."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        advisor = FinancialAdvisor(quantization="none")

        # Mock analyze to fail on second query
        def mock_analyze(query, **kwargs):
            if query == "Query 2":
                raise Exception("Test error")
            return {
                "analysis": f"Analysis for {query}",
                "confidence": 0.8,
                "entities": [],
                "latency_ms": 100.0,
            }

        advisor.analyze = Mock(side_effect=mock_analyze)

        queries = ["Query 1", "Query 2", "Query 3"]
        results = advisor.batch_analyze(queries)

        assert len(results) == 3
        assert "Error" in results[1]["analysis"]


class TestFactoryFunction:
    """Test factory function."""

    @patch("src.inference.AutoModelForCausalLM")
    @patch("src.inference.AutoTokenizer")
    def test_create_advisor(self, mock_tokenizer_class, mock_model_class):
        """Test factory function creates advisor."""
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        advisor = create_advisor(quantization="none")

        assert isinstance(advisor, FinancialAdvisor)


class TestUtilsImport:
    """Test utility function imports."""

    def test_extract_entities_import(self):
        """Test extract_entities can be imported from utils."""
        from src.utils import extract_entities, flatten_entities

        text = "Apple Inc. reported $89.5 billion revenue."
        entities = extract_entities(text)

        assert isinstance(entities, dict)
        assert "company" in entities
        assert "money" in entities

    def test_create_prompt_import(self):
        """Test create_prompt can be imported from utils."""
        from src.utils import create_prompt

        prompt = create_prompt(
            user_query="What is Apple's stock price?",
            system_prompt="You are a financial advisor.",
            format_type="phi"
        )

        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt


class TestConfigIntegration:
    """Test configuration integration."""

    def test_config_loading(self):
        """Test config can be loaded for inference."""
        from src.config import Config

        config = Config()

        assert config.model.name == "microsoft/phi-3.5-mini-instruct"
        assert config.generation.temperature == 0.7
        assert config.generation.top_p == 0.95
