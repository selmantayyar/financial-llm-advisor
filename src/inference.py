

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from contextlib import asynccontextmanager
import time
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from .utils import extract_entities, flatten_entities, create_prompt

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL_NAME = "microsoft/phi-3.5-mini-instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a financial advisor assistant with expertise in investment analysis, "
    "portfolio management, and financial markets. You provide clear, data-driven "
    "investment insights with proper reasoning and confidence levels."
)


class AnalysisRequest(BaseModel):

    query: str = Field(..., description="Investment question to analyze")
    context: Optional[str] = Field(None, description="Optional document context")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling probability")


class AnalysisResponse(BaseModel):

    analysis: str = Field(..., description="Generated analysis text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: List[str] = Field(default_factory=list, description="Extracted financial entities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchRequest(BaseModel):

    queries: List[str] = Field(..., min_length=1, description="List of queries to analyze")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens per response")


class HealthResponse(BaseModel):

    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device used for inference")


class FinancialAdvisor:

    def __init__(
        self,
        base_model: str = DEFAULT_MODEL_NAME,
        lora_weights: Optional[str] = None,
        quantization: str = "8bit",
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):

        self.base_model = base_model
        self.lora_weights = lora_weights
        self.quantization = quantization
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        # Generation configuration
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True,
        }

        # Load model on initialization
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model: {self.base_model}")
        logger.info(f"Quantization: {self.quantization}")

        # Configure quantization
        if self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        elif self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not quantization_config else None,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load LoRA adapter if provided
        if self.lora_weights:
            logger.info(f"Loading LoRA adapter from: {self.lora_weights}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_weights,
            )

        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")

    def merge_lora(self, output_path: Optional[str] = None) -> None:
        if not isinstance(self.model, PeftModel):
            logger.warning("Model is not a PEFT model, nothing to merge")
            return

        logger.info("Merging LoRA adapter into base model...")
        self.model = self.model.merge_and_unload()

        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            logger.info(f"Merged model saved to {output_path}")

    def _compute_confidence(self, output_ids: torch.Tensor, scores: List[torch.Tensor]) -> float:

        if not scores:
            return 0.5

        # Get probabilities for generated tokens
        probs = []
        for i, score in enumerate(scores):
            if i >= len(output_ids) - 1:
                break
            token_id = output_ids[i + 1].item()
            softmax = torch.softmax(score[0], dim=-1)
            prob = softmax[token_id].item()
            probs.append(prob)

        if not probs:
            return 0.5

        # Use geometric mean of probabilities as confidence
        log_probs = [np.log(p + 1e-10) for p in probs]
        mean_log_prob = np.mean(log_probs)
        confidence = np.exp(mean_log_prob)

        # Normalize to [0, 1]
        confidence = min(max(confidence, 0.0), 1.0)

        return float(confidence)

    def analyze(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Create formatted prompt
        prompt = create_prompt(
            user_query=query,
            system_prompt=self.system_prompt,
            context=context,
            format_type="phi",
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate with timing
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                temperature=temperature or self.generation_config["temperature"],
                top_p=top_p or self.generation_config["top_p"],
                top_k=self.generation_config["top_k"],
                repetition_penalty=self.generation_config["repetition_penalty"],
                do_sample=self.generation_config["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Decode output
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        analysis = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean up response (remove any trailing special tokens)
        if "<|end|>" in analysis:
            analysis = analysis.split("<|end|>")[0].strip()

        confidence = self._compute_confidence(outputs.sequences[0], list(outputs.scores))

        # Extract entities
        entity_dict = extract_entities(analysis)
        entities = flatten_entities(entity_dict)

        return {
            "analysis": analysis,
            "confidence": confidence,
            "entities": entities,
            "latency_ms": latency_ms,
        }

    def batch_analyze(
        self,
        queries: List[str],
        max_tokens: int = 512,
    ) -> List[Dict[str, Any]]:

        results = []
        for query in queries:
            try:
                result = self.analyze(query, max_tokens=max_tokens)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing query: {e}")
                results.append({
                    "analysis": f"Error: {str(e)}",
                    "confidence": 0.0,
                    "entities": [],
                    "latency_ms": 0.0,
                })
        return results


# Global advisor instance
advisor: Optional[FinancialAdvisor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global advisor
    # Startup: Load model
    logger.info("Starting Financial LLM Advisor API...")
    try:
        advisor = FinancialAdvisor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        advisor = None
    yield
    # Shutdown
    logger.info("Shutting down...")


# FastAPI app setup
app = FastAPI(
    title="Financial LLM Advisor API",
    description="Production-grade investment analysis API powered by fine-tuned Phi-3.5-mini",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:

    global advisor

    if advisor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = advisor.analyze(
            query=request.query,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return AnalysisResponse(
            analysis=result["analysis"],
            confidence=result["confidence"],
            entities=result["entities"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
async def batch(request: BatchRequest) -> List[AnalysisResponse]:

    global advisor

    if advisor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = advisor.batch_analyze(
            queries=request.queries,
            max_tokens=request.max_tokens,
        )

        return [
            AnalysisResponse(
                analysis=r["analysis"],
                confidence=r["confidence"],
                entities=r["entities"],
                latency_ms=r["latency_ms"],
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:

    global advisor

    return HealthResponse(
        status="ok" if advisor is not None else "degraded",
        model_loaded=advisor is not None,
        device=advisor.device if advisor else "unknown",
    )


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "Financial LLM Advisor API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


def create_advisor(
    base_model: str = DEFAULT_MODEL_NAME,
    lora_weights: Optional[str] = None,
    **kwargs,
) -> FinancialAdvisor:

    return FinancialAdvisor(
        base_model=base_model,
        lora_weights=lora_weights,
        **kwargs,
    )
