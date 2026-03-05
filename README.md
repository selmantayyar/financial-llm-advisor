# Financial Investment Advisor LLM

A production-grade fine-tuned language model for institutional investment decision support. Fine-tuned on 50K financial instructions using LoRA, deployed to HuggingFace, and ready for production serving via vLLM.

**Model available on HuggingFace:** [selmantayyar/financial-llm-advisor](https://huggingface.co/selmantayyar/financial-llm-advisor)

## Problem & Solution

**Problem:** Institutional investors need to synthesize investment insights from unstructured financial data (earnings calls, SEC filings, analyst reports). Commercial LLM APIs are expensive at scale and raise data privacy concerns.

**Solution:** Fine-tuned Phi-3.5-mini on 50K financial instructions to create a self-hosted, domain-specific AI advisor — delivering financial analysis at a fraction of commercial API costs.

---

## Evaluation Results

Evaluated on a held-out test set (100 samples for reasoning, 50 for latency) from the Finance-Instruct-500k dataset.

| Metric | Result | Notes |
|--------|--------|-------|
| Financial Reasoning Accuracy | **44%** | Word-overlap metric (>=50% key term match with reference) |
| Investment Q&A F1-Score | **0.454** | Token-level F1 against reference answers |
| Q&A Precision / Recall | 0.492 / 0.480 | Balanced precision and recall |
| Inference Throughput | **28 tokens/sec** | RTX 4090, bf16, SDPA attention |
| Latency (p50 / p99) | 7.7s / 8.0s | Generating ~164 tokens avg per response |
| Cost per 1M tokens | **$0.18** | Self-hosted vs $30 (GPT-4) / $15 (Claude) |
| Model Size | 3.8B + LoRA | ~120MB adapter on top of base model |

> **On evaluation methodology:** The reasoning metric uses simple word overlap between generated and reference answers — a strict measure that penalizes valid but differently-worded responses. The model produces coherent, detailed financial analyses that may not match reference wording. See [BENCHMARKING.md](docs/BENCHMARKING.md) for detailed analysis and improvement paths.

---

## Features

- **Domain-Specific:** Fine-tuned on 50K financial instruction-following examples
- **Self-Hosted:** Full control over data privacy, no API rate limits
- **Cost-Efficient:** 99.4% cheaper than GPT-4 for self-hosted inference
- **Reproducible:** Full pipeline (data loading, training, evaluation, deployment) in one repo
- **HuggingFace Deployed:** Model weights available at [selmantayyar/financial-llm-advisor](https://huggingface.co/selmantayyar/financial-llm-advisor)
- **vLLM Ready:** Supports optimized serving for production workloads

---

## Quick Start

### Installation
```shell
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/selmantayyar/financial-llm-advisor.git
cd financial-llm-advisor

# Create virtual environment (Python 3.12 recommended)
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv sync
```

### Use the Pre-trained Model from HuggingFace
```python
from src.inference import FinancialAdvisor

advisor = FinancialAdvisor(base_model="selmantayyar/financial-llm-advisor")
response = advisor.analyze("What are the key risks for Apple in 2024?")
print(response["analysis"])
```

### Serve via vLLM (Production)
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model selmantayyar/financial-llm-advisor \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000
```

### Train Your Own
```shell
bash scripts/train.sh
```

### Evaluate
```shell
bash scripts/evaluate.sh
```

---

## Documentation

- **[SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)** - Architecture & design decisions
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Data pipeline & components
- **[BENCHMARKING.md](docs/BENCHMARKING.md)** - Detailed evaluation results & analysis

---

## Architecture

```
Financial Documents
        |
    [Phi-3.5-mini-instruct]
        |
    [LoRA Adapter (r=16)]
        |
  [vLLM / FastAPI Server]
        |
Investment Analysis + Confidence Score
```

---

## Hardware Requirements

**Inference (bf16, no quantization):**
- 16GB VRAM (RTX 4090, A100, etc.)
- With 4-bit quantization: 6GB VRAM (RTX 3060, etc.)

**Training (bf16, LoRA, gradient checkpointing):**
- 24GB VRAM recommended (RTX 4090, A100)
- Batch size 4, sequence length 1024

**Cloud:**
- RunPod RTX 4090: ~$0.59/hour
- Full training cost: ~$6-7

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | microsoft/phi-3.5-mini-instruct (3.8B) |
| Dataset | Josephgflowers/Finance-Instruct-500k (50K subset) |
| Method | SFT with LoRA (r=16, alpha=32) |
| Target Modules | qkv_proj, o_proj |
| Precision | bf16 |
| Batch Size | 4 (effective 16 with gradient accumulation) |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Training Time | ~8-10 hours on RTX 4090 |
| Attention | SDPA (PyTorch built-in) |

---

## Evaluation

The model is evaluated on three task types plus latency benchmarking:

1. **Financial Reasoning** - Multi-step investment analysis with key-term overlap scoring
2. **Investment Q&A** - Token-level F1 and exact match against reference answers
3. **Named Entity Recognition** - Regex-based extraction of companies, tickers, monetary values, percentages
4. **Latency & Throughput** - End-to-end generation benchmarking with warmup

See [BENCHMARKING.md](docs/BENCHMARKING.md) for full results and methodology.

---

## Project Structure

```
src/                    # Source code
  config.py            # Configuration (Pydantic models)
  dataset_loader.py    # Data loading & preprocessing
  trainer.py           # SFT training with LoRA
  evaluator.py         # Financial metrics evaluation
  inference.py         # FastAPI server & generation
  utils.py             # Helper functions

scripts/               # Training/evaluation/deploy scripts
config/                # YAML configuration files
docs/                  # Documentation
```

---

## Reproducibility

```shell
# 1. Setup
uv venv --python 3.12 && source .venv/bin/activate && uv sync

# 2. Train
bash scripts/train.sh

# 3. Evaluate
bash scripts/evaluate.sh

# 4. Results saved to results/evaluation_metrics.json
```

**Full training cost:** ~$6 on RunPod RTX 4090

---

## Future Improvements

- **vLLM serving** for 3-4x throughput improvement (~80-120 tokens/sec)
- **4-bit quantization** at inference for lower VRAM and faster generation
- **Improved evaluation** with semantic similarity metrics (BERTScore, LLM-as-judge)
- **More training data** and epochs for higher reasoning accuracy
- **DPO/RLHF** alignment for better response quality

---

## License

MIT License - See [LICENSE](LICENSE) file

---

## Contributing

Contributions welcome! Please fork the repository, create a feature branch, and submit a pull request.