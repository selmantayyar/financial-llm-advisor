#!/bin/bash
# Training script for Financial LLM Advisor
# Usage: bash scripts/train.sh [config_path] [output_dir]

set -e

echo "========================================"
echo "  Financial LLM Advisor Training"
echo "========================================"

# Configuration
CONFIG_PATH="${1:-config/training_config.yaml}"
OUTPUT_DIR="${2:-./checkpoints}"
PYTHON_EXEC=.venv/bin/python

# Check if Python is available
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.9+"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo ""
echo "Configuration:"
echo "  Config path: $CONFIG_PATH"
echo "  Output dir:  $OUTPUT_DIR"
echo ""

# Run training
$PYTHON_EXEC << 'PYTHON_SCRIPT'
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, '.')

from src.config import Config
from src.dataset_loader import DatasetLoader
from src.trainer import FinancialLLMTrainer

def main():
    config_path = os.environ.get('CONFIG_PATH', 'config/training_config.yaml')
    output_dir = os.environ.get('OUTPUT_DIR', './checkpoints')

    logger.info("Loading configuration...")
    config = Config(config_path)

    logger.info("="*50)
    logger.info("Configuration Summary:")
    logger.info(f"  Model: {config.model.name}")
    logger.info(f"  Dataset: {config.dataset.name}")
    logger.info(f"  Subset size: {config.dataset.subset_size}")
    logger.info(f"  LoRA rank: {config.training.lora.rank}")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info("="*50)

    # Load dataset
    logger.info("Loading and processing dataset...")
    loader = DatasetLoader(config.dataset.model_dump())
    train_ds, val_ds = loader.load_and_process_dataset()

    logger.info(f"Train dataset: {len(train_ds)} examples")
    logger.info(f"Val dataset: {len(val_ds)} examples")

    # Initialize trainer
    logger.info("Initializing trainer...")
    training_config = {
        "model_name": config.model.name,
        "lora_rank": config.training.lora.rank,
        "lora_alpha": config.training.lora.alpha,
        "lora_dropout": config.training.lora.dropout,
        "target_modules": config.training.lora.target_modules,
        "learning_rate": config.training.learning_rate,
        "num_epochs": config.training.num_epochs,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "warmup_steps": config.training.warmup_steps,
        "logging_steps": config.training.logging_steps,
        "eval_steps": config.training.eval_steps,
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "output_dir": output_dir,
        "wandb_project": config.training.wandb.project,
        "wandb_enabled": config.training.wandb.enabled,
        "seed": config.training.seed,
        "dataloader_num_workers": config.training.dataloader_num_workers,
    }

    trainer = FinancialLLMTrainer(training_config)

    # Load model
    logger.info("Loading base model...")
    trainer.load_model()

    # Setup LoRA
    logger.info("Setting up LoRA adapters...")
    trainer.setup_lora()

    # Train
    logger.info("Starting training...")
    logger.info("="*50)
    metrics = trainer.train(train_ds, val_ds)
    logger.info("="*50)

    # Save final model
    logger.info("Saving final model...")
    model_path = trainer.save_model(f"{output_dir}/final_model")

    logger.info("="*50)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo ""
echo "========================================"
echo "  Training Complete!"
echo "  Model saved to: $OUTPUT_DIR/final_model"
echo "========================================"
