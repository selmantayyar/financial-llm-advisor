"""Financial LLM Advisor - Production-grade fine-tuned model for investment analysis."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.config import Config
from src.dataset_loader import DatasetLoader
from src.trainer import FinancialLLMTrainer
from src.evaluator import FinancialLLMEvaluator
from src.inference import FinancialAdvisor

__all__ = [
    "Config",
    "DatasetLoader",
    "FinancialLLMTrainer",
    "FinancialLLMEvaluator",
    "FinancialAdvisor",
]