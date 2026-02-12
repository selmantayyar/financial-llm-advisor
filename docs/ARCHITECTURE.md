# Architecture: Financial LLM Advisor

## System Overview
```
┌─────────────────────────────────────────────────────┐
│         Financial Documents & Queries                │
│  (earnings calls, SEC filings, analyst reports)     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │  Data Preprocessing  │
        │  - Tokenization      │
        │  - Length filtering  │
        │  - Format handling   │
        └──────────┬───────────┘
                   │
                   ↓
    ┌──────────────────────────────────┐
    │    Phi-3.5-mini Foundation       │
    │    (3.8B parameters, fp32)       │
    └──────────┬───────────────────────┘
               │
               ↓
    ┌──────────────────────────────────┐
    │       LoRA Adapter Layer         │
    │  (1.2M trainable params, r=16)   │
    └──────────┬───────────────────────┘
               │
               ├─── [Training Mode] ──→ Gradient computation
               │
               └─── [Inference Mode] → Token generation
                                       ↓
                            ┌──────────────────────┐
                            │   Post-processing    │
                            │  - Temperature       │
                            │  - Top-p sampling    │
                            │  - Confidence score  │
                            └──────────┬───────────┘
                                       │
                                       ↓
                        ┌──────────────────────────┐
                        │  Investment Analysis     │
                        │  with Reasoning          │
                        └──────────────────────────┘
```

## Data Pipeline

### Input Stage
```python
# Raw data from Hugging Face
dataset = load_dataset("Josephgflowers/Finance-Instruct-500k")

# Filter & clean
- Remove examples < 50 tokens (too simple)
- Remove examples > 1000 tokens (too long)
- Remove invalid JSON/formatting

# Stratified split
- Train: 40K (80%)
- Validation: 5K (10%)
- Test: 5K (10%)
```

### Processing Stage
```python
# Tokenization
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3.5-mini-instruct")
encoded = tokenizer(
    text,
    max_length=512,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

# Batching
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

## Training Pipeline

### Initialization
```
Model Setup:
├─ Load base model (Phi-3.5-mini)
├─ Apply 8-bit quantization (reduces memory)
├─ Freeze base model weights
├─ Initialize LoRA adapters (r=16, α=32)
└─ Setup optimizer (AdamW)

Training Args:
├─ num_epochs = 3
├─ batch_size = 16 per device
├─ learning_rate = 2e-4
├─ warmup_steps = 500
├─ save_steps = 500
└─ eval_steps = 500
```

### Training Loop
```
For each epoch:
  For each batch:
    1. Forward pass
    2. Compute loss (cross-entropy)
    3. Backward pass (only LoRA grads)
    4. Update LoRA weights
    5. Log metrics (loss, learning rate)
    6. Save checkpoint every 500 steps
    7. Evaluate on val set every 500 steps

Expected:
- Total steps: 7,500 (40K examples / 16 batch size * 3 epochs)
- Checkpoints: 15
- Training time: 5-6 hours on RTX 4090
```

### Monitoring with Weights & Biases
```
Tracked Metrics:
├─ Training loss (should decrease)
├─ Validation loss (monitor for overfitting)
├─ Learning rate (schedule)
├─ GPU memory usage
├─ Training speed (tokens/sec)
└─ Evaluation metrics (accuracy, F1)

Dashboard: https://wandb.ai/your-username/financial-llm-advisor
```

## Evaluation Pipeline

### Metrics Computed
```
1. Financial Reasoning Accuracy
   - Multi-step analysis tasks
   - Evaluate model output vs reference
   - Metric: Exact match %

2. Investment Q&A F1-Score
   - Answer spans in text
   - Overlap with reference answer
   - Metric: Macro F1 (0-1)

3. Named Entity Recognition
   - Extract financial entities (companies, people, regulations)
   - Compare extracted spans
   - Metric: Seqeval F1

4. Inference Metrics
   - Latency (p50, p99)
   - Throughput (tokens/sec)
   - Memory usage
   - Cost estimation
```

### Evaluation Code Structure
```python
# src/evaluator.py
class FinancialLLMEvaluator:
    def evaluate_reasoning(self) -> float:
        """Accuracy on multi-step financial analysis"""
    
    def evaluate_qa(self) -> Dict[str, float]:
        """F1-score on investment Q&A"""
    
    def evaluate_ner(self) -> Dict[str, float]:
        """F1-score on entity recognition"""
    
    def evaluate_latency(self) -> Dict[str, float]:
        """Inference speed benchmarks"""
```

## Inference Pipeline

### Model Loading
```python
from src.inference import FinancialAdvisor

# Initialize (5-10 seconds)
advisor = FinancialAdvisor(
    base_model="microsoft/phi-3.5-mini-instruct",
    lora_weights="path/to/adapter_model.safetensors",
    quantization="8bit"
)

# Merge LoRA into base model (optional)
advisor.merge_lora()  # Single model file for production
```

### Generation Parameters
```python
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,  # Deterministic but creative
    "top_p": 0.95,       # Nucleus sampling
    "top_k": 50,
    "repetition_penalty": 1.0,
}
```

### Example Workflow
```python
# User query
question = "What are the key risks for Apple in 2024?"

# Generate response (takes ~200ms)
response = advisor.analyze(
    question,
    document_context="[earnings call transcript]"  # Optional
)

# Output includes:
# - analysis: str  (investment reasoning)
# - confidence: float  (0-1, how certain model is)
# - entities: List[str]  (extracted companies, people)
# - references: List[str]  (which parts of document used)
```

## API Server

### FastAPI Endpoints
```python
# src/inference.py (with FastAPI)

@app.post("/analyze")
async def analyze(
    query: str,
    context: Optional[str] = None,
    max_tokens: int = 512
) -> Dict:
    """Analyze investment question"""
    response = advisor.analyze(query, context, max_tokens)
    return response

@app.get("/health")
async def health() -> Dict:
    """Health check"""
    return {"status": "ok"}

@app.post("/batch")
async def batch_analyze(
    queries: List[str]
) -> List[Dict]:
    """Batch processing"""
    return [advisor.analyze(q) for q in queries]
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
COPY pyproject.toml .
RUN uv sync

COPY src/ src/
COPY config/ config/

CMD ["python", "-m", "uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Characteristics

### Memory Usage
```
Model components:
├─ Base model (fp8): 1.9 GB
├─ LoRA adapter: 120 MB
├─ Tokenizer: 200 MB
├─ Generation buffer: 2-4 GB (batch_size dependent)
└─ Total: ~5 GB

With 8-bit quantization: ~2-3 GB for inference
With 4-bit quantization: ~1-2 GB for inference (slower)
```

### Latency Breakdown
```
Query: "What are key risks for Apple?"

Components:
├─ Tokenization: 5ms (string → tokens)
├─ Model loading (cold start): 2 seconds (first call)
├─ Model inference: 180ms (token generation)
├─ Post-processing: 10ms (format response)
└─ Total (warm): ~200ms
   Total (cold): ~2.2 seconds

Throughput:
├─ Single GPU: 100+ requests/sec
├─ With batching: 500+ requests/sec
└─ CPU: 5-10 requests/sec
```

### Cost Estimation
```
Per 1M tokens:
├─ Model: 3.8B params (amortized)
├─ Inference (GPU): $0.0035
├─ Inference (CPU): $0.0001
├─ Storage: negligible
└─ Total: ~$0.004 per 1M tokens

vs GPT-4: $0.03 per 1M tokens
Savings: 7.5x cheaper
```

## Monitoring & Logging

### What We Log
```
├─ Request metrics
│  ├─ Input length
│  ├─ Output length
│  ├─ Latency
│  └─ Error rate
│
├─ Model metrics
│  ├─ GPU memory
│  ├─ Cache size
│  ├─ Batch size
│  └─ Temperature
│
└─ Business metrics
   ├─ Daily requests
   ├─ Avg response quality
   ├─ Cost tracking
   └─ User feedback
```

### Logging Implementation
```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/advisor.log")
logger.addHandler(file_handler)

# Structured logging for production
import json
logger.info(json.dumps({
    "timestamp": datetime.now().isoformat(),
    "query_length": len(query),
    "response_latency_ms": latency,
    "confidence": confidence,
}))
```

---

This architecture is designed for:
- ✅ Production deployment (scalable, monitored)
- ✅ Research iteration (easy to experiment)
- ✅ Cost efficiency (run on consumer hardware)
- ✅ Reproducibility (documented, version-controlled)