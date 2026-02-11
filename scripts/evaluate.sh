#!/bin/bash
# Evaluation script for Financial LLM Advisor
# Usage: bash scripts/evaluate.sh [model_path] [output_path] [num_samples]

set -e

echo "========================================"
echo "  Financial LLM Advisor Evaluation"
echo "========================================"

# Configuration
MODEL_PATH="${1:-./checkpoints/final_model}"
OUTPUT_PATH="${2:-results/evaluation_metrics.json}"
NUM_SAMPLES="${3:-100}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"

# Check if Python is available
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.9+"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p logs

echo ""
echo "Configuration:"
echo "  Model path:   $MODEL_PATH"
echo "  Output path:  $OUTPUT_PATH"
echo "  Num samples:  $NUM_SAMPLES"
echo ""

# Export for Python script
export MODEL_PATH
export OUTPUT_PATH
export NUM_SAMPLES

# Run evaluation
$PYTHON_EXEC << 'PYTHON_SCRIPT'
import sys
import os
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, '.')

from src.inference import FinancialAdvisor
from src.evaluator import FinancialLLMEvaluator
from src.dataset_loader import DatasetLoader

def main():
    model_path = os.environ.get('MODEL_PATH', './checkpoints/final_model')
    output_path = os.environ.get('OUTPUT_PATH', 'results/evaluation_metrics.json')
    num_samples = int(os.environ.get('NUM_SAMPLES', '100'))

    logger.info("="*50)
    logger.info("Starting Evaluation")
    logger.info("="*50)

    # Load model
    logger.info(f"Loading model from: {model_path}")
    try:
        advisor = FinancialAdvisor(lora_weights=model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load LoRA weights: {e}")
        logger.info("Loading base model without LoRA...")
        advisor = FinancialAdvisor()

    # Load test dataset
    logger.info("Loading test dataset...")
    loader = DatasetLoader({})
    test_ds = loader.load_test_dataset()
    logger.info(f"Test dataset size: {len(test_ds)}")

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = FinancialLLMEvaluator(advisor.model, advisor.tokenizer)

    # Run evaluation
    logger.info(f"Running evaluation on {num_samples} samples...")
    logger.info("="*50)
    metrics = evaluator.evaluate_all(test_ds, num_samples=num_samples)
    logger.info("="*50)

    # Save metrics
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {output_path}")

    # Print summary
    logger.info("")
    logger.info("="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Reasoning Accuracy: {metrics.get('reasoning_accuracy', 0):.1%}")
    logger.info(f"Q&A F1-Score:       {metrics.get('qa_f1', 0):.3f}")
    logger.info(f"NER F1-Score:       {metrics.get('ner_f1', 0):.3f}")
    logger.info(f"Latency (p99):      {metrics.get('latency_p99', 0):.1f}ms")
    logger.info(f"Cost per 1M tokens: ${metrics.get('cost_per_million_tokens', 0):.2f}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo ""
echo "========================================"
echo "  Evaluation Complete!"
echo "  Results saved to: $OUTPUT_PATH"
echo "========================================"
