"""Evaluation metrics for Financial LLM Advisor.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from collections import Counter
import time
import re
import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

# Cost per 1M tokens for self-hosted inference
COST_PER_MILLION_TOKENS = 0.18  # USD


class FinancialLLMEvaluator:

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_config = generation_config or {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
        }

        # Move model to device if needed
        if hasattr(self.model, "to") and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.device)

        self.model.eval()

    def _generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, float]:

        max_new_tokens = max_new_tokens or self.generation_config.get("max_new_tokens", 256)

        # Tokenize input
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
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.generation_config.get("temperature", 0.7),
                top_p=self.generation_config.get("top_p", 0.95),
                do_sample=self.generation_config.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return generated_text.strip(), latency_ms

    def _normalize_text(self, text: str) -> str:

        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def _compute_exact_match(self, prediction: str, reference: str) -> float:

        pred_normalized = self._normalize_text(prediction)
        ref_normalized = self._normalize_text(reference)
        return 1.0 if pred_normalized == ref_normalized else 0.0

    def _compute_token_f1(self, prediction: str, reference: str) -> Dict[str, float]:
        pred_tokens = self._normalize_text(prediction).split()
        ref_tokens = self._normalize_text(reference).split()

        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        # Count common tokens
        common = sum((pred_counter & ref_counter).values())

        # Compute precision and recall
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(ref_tokens) if ref_tokens else 0.0

        # Compute F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_reasoning(
        self,
        test_dataset: Dataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:

        logger.info("Evaluating financial reasoning accuracy...")

        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        correct = 0
        total = 0
        latencies = []

        for i, example in enumerate(test_dataset):
            # Extract prompt and reference
            if "instruction" in example:
                prompt = str(example.get("instruction", ""))
                if example.get("input"):
                    prompt += f"\n\nContext: {example['input']}"
                reference = str(example.get("output", ""))
            else:
                prompt = str(example.get("question", example.get("prompt", "")))
                reference = str(example.get("answer", example.get("response", "")))

            if not prompt or not reference:
                continue

            # Format prompt
            formatted_prompt = (
                f"<|system|>\nYou are a financial advisor. Provide a clear, accurate analysis.<|end|>\n"
                f"<|user|>\n{prompt}<|end|>\n"
                f"<|assistant|>\n"
            )

            # Generate response
            try:
                prediction, latency = self._generate_response(formatted_prompt)
                latencies.append(latency)

                # Check if key concepts match
                ref_key_terms = set(self._normalize_text(reference).split())
                pred_key_terms = set(self._normalize_text(prediction).split())

                # Calculate overlap ratio
                overlap = len(ref_key_terms & pred_key_terms) / len(ref_key_terms) if ref_key_terms else 0

                if overlap >= 0.5:  # At least 50% of key terms match
                    correct += 1

                total += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_dataset)} samples")

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0

        results = {
            "reasoning_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_latency_ms": np.mean(latencies) if latencies else 0.0,
        }

        logger.info(f"Reasoning accuracy: {accuracy:.2%}")
        return results

    def evaluate_qa(
        self,
        test_dataset: Dataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:

        logger.info("Evaluating Q&A F1-score...")

        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        all_f1_scores = []
        all_precision = []
        all_recall = []
        exact_matches = 0
        total = 0

        for i, example in enumerate(test_dataset):
            # Extract question and reference answer
            if "instruction" in example:
                question = str(example.get("instruction", ""))
                reference = str(example.get("output", ""))
            else:
                question = str(example.get("question", example.get("prompt", "")))
                reference = str(example.get("answer", example.get("response", "")))

            if not question or not reference:
                continue

            # Format prompt
            formatted_prompt = (
                f"<|system|>\nYou are a financial advisor. Answer questions accurately.<|end|>\n"
                f"<|user|>\n{question}<|end|>\n"
                f"<|assistant|>\n"
            )

            try:
                prediction, _ = self._generate_response(formatted_prompt)

                # Compute metrics
                f1_metrics = self._compute_token_f1(prediction, reference)
                all_f1_scores.append(f1_metrics["f1"])
                all_precision.append(f1_metrics["precision"])
                all_recall.append(f1_metrics["recall"])

                # Check exact match
                if self._compute_exact_match(prediction, reference) == 1.0:
                    exact_matches += 1

                total += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_dataset)} samples")

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        results = {
            "qa_f1": np.mean(all_f1_scores) if all_f1_scores else 0.0,
            "qa_precision": np.mean(all_precision) if all_precision else 0.0,
            "qa_recall": np.mean(all_recall) if all_recall else 0.0,
            "qa_exact_match": exact_matches / total if total > 0 else 0.0,
            "total_samples": total,
        }

        logger.info(f"Q&A F1-score: {results['qa_f1']:.3f}")
        return results

    def evaluate_ner(
        self,
        test_dataset: Dataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:

        logger.info("Evaluating Named Entity Recognition...")

        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        # Entity extraction patterns
        entity_patterns = {
            "company": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|Corp\.|Company|Co\.|Ltd\.)\b",
            "ticker": r"\b[A-Z]{1,5}\b(?=\s+(?:stock|shares))",
            "money": r"\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|B|M)?",
            "percentage": r"[\d.]+\s*%",
        }

        all_precision = []
        all_recall = []
        all_f1 = []

        for i, example in enumerate(test_dataset):
            # Get text with entities
            if "text" in example:
                text = str(example["text"])
            elif "output" in example:
                text = str(example["output"])
            else:
                continue

            # Extract expected entities
            reference_entities = set()
            for pattern in entity_patterns.values():
                matches = re.findall(pattern, text, re.IGNORECASE)
                reference_entities.update(matches)

            if not reference_entities:
                continue

            # Create NER prompt
            ner_prompt = (
                f"<|system|>\nExtract all financial entities from the text.<|end|>\n"
                f"<|user|>\nExtract companies, tickers, monetary values, and percentages from: {text[:500]}<|end|>\n"
                f"<|assistant|>\n"
            )

            try:
                prediction, _ = self._generate_response(ner_prompt)

                # Extract predicted entities
                predicted_entities = set()
                for pattern in entity_patterns.values():
                    matches = re.findall(pattern, prediction, re.IGNORECASE)
                    predicted_entities.update(matches)

                # Compute metrics
                if predicted_entities and reference_entities:
                    true_positives = len(predicted_entities & reference_entities)
                    precision = true_positives / len(predicted_entities)
                    recall = true_positives / len(reference_entities)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    precision = recall = f1 = 0.0

                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_dataset)} samples")

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        results = {
            "ner_f1": np.mean(all_f1) if all_f1 else 0.0,
            "ner_precision": np.mean(all_precision) if all_precision else 0.0,
            "ner_recall": np.mean(all_recall) if all_recall else 0.0,
            "total_samples": len(all_f1),
        }

        logger.info(f"NER F1-score: {results['ner_f1']:.3f}")
        return results

    def evaluate_latency(
        self,
        num_samples: int = 100,
        warmup_samples: int = 5,
    ) -> Dict[str, float]:

        logger.info(f"Evaluating latency with {num_samples} samples...")

        # Test prompts of varying lengths
        test_prompts = [
            "What is the current stock price of Apple?",
            "Analyze the risk factors for investing in technology stocks.",
            "Compare the P/E ratios of major tech companies and provide investment recommendations.",
            "Explain the impact of interest rate changes on bond prices and suggest portfolio adjustments.",
            "Provide a comprehensive analysis of the semiconductor industry including key players, market trends, and future outlook.",
        ]

        latencies = []

        # Warmup
        logger.info(f"Running {warmup_samples} warmup samples...")
        for _ in range(warmup_samples):
            prompt = test_prompts[0]
            formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            self._generate_response(formatted)

        # Benchmark
        for i in range(num_samples):
            prompt = test_prompts[i % len(test_prompts)]
            formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

            try:
                _, latency = self._generate_response(formatted)
                latencies.append(latency)

                if (i + 1) % 20 == 0:
                    logger.info(f"Benchmark progress: {i + 1}/{num_samples}")

            except Exception as e:
                logger.warning(f"Error in sample {i}: {e}")
                continue

        if not latencies:
            return {
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "latency_p99": 0.0,
                "latency_mean": 0.0,
                "latency_std": 0.0,
            }

        latencies = np.array(latencies)

        results = {
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "latency_mean": float(np.mean(latencies)),
            "latency_std": float(np.std(latencies)),
            "latency_min": float(np.min(latencies)),
            "latency_max": float(np.max(latencies)),
            "num_samples": len(latencies),
        }

        logger.info(f"Latency p50: {results['latency_p50']:.1f}ms, p99: {results['latency_p99']:.1f}ms")
        return results

    def estimate_cost(
        self,
        num_tokens: int = 1_000_000,
    ) -> Dict[str, float]:

        # Self-hosted cost estimate
        self_hosted_cost = (num_tokens / 1_000_000) * COST_PER_MILLION_TOKENS

        # Cloud API comparison costs (approximate)
        gpt4_cost = (num_tokens / 1_000_000) * 30.0  # GPT-4 pricing
        claude_cost = (num_tokens / 1_000_000) * 15.0  # Claude pricing

        results = {
            "cost_per_million_tokens": COST_PER_MILLION_TOKENS,
            "self_hosted_cost": self_hosted_cost,
            "gpt4_equivalent_cost": gpt4_cost,
            "claude_equivalent_cost": claude_cost,
            "savings_vs_gpt4": gpt4_cost - self_hosted_cost,
            "savings_vs_claude": claude_cost - self_hosted_cost,
            "savings_percentage_vs_gpt4": (1 - self_hosted_cost / gpt4_cost) * 100 if gpt4_cost > 0 else 0,
        }

        logger.info(f"Cost per 1M tokens: ${COST_PER_MILLION_TOKENS:.2f}")
        return results

    def evaluate_all(
        self,
        test_dataset: Dataset,
        num_samples: Optional[int] = None,
        skip_latency: bool = False,
    ) -> Dict[str, Any]:

        logger.info("Starting comprehensive evaluation...")

        results = {}

        # Reasoning evaluation
        logger.info("=" * 50)
        logger.info("1. Evaluating Financial Reasoning")
        reasoning_results = self.evaluate_reasoning(test_dataset, num_samples)
        results.update(reasoning_results)

        # Q&A evaluation
        logger.info("=" * 50)
        logger.info("2. Evaluating Q&A Performance")
        qa_results = self.evaluate_qa(test_dataset, num_samples)
        results.update(qa_results)

        # NER evaluation
        logger.info("=" * 50)
        logger.info("3. Evaluating Named Entity Recognition")
        ner_results = self.evaluate_ner(test_dataset, num_samples)
        results.update(ner_results)

        # Latency evaluation
        if not skip_latency:
            logger.info("=" * 50)
            logger.info("4. Evaluating Inference Latency")
            latency_results = self.evaluate_latency(num_samples=50)
            results.update(latency_results)

        # Cost estimation
        logger.info("=" * 50)
        logger.info("5. Estimating Costs")
        cost_results = self.estimate_cost()
        results.update(cost_results)

        # Summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Reasoning Accuracy: {results.get('reasoning_accuracy', 0):.1%}")
        logger.info(f"Q&A F1-Score: {results.get('qa_f1', 0):.3f}")
        logger.info(f"NER F1-Score: {results.get('ner_f1', 0):.3f}")
        if not skip_latency:
            logger.info(f"Latency (p99): {results.get('latency_p99', 0):.1f}ms")
        logger.info(f"Cost per 1M tokens: ${results.get('cost_per_million_tokens', 0):.2f}")

        return results


def create_evaluator(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> FinancialLLMEvaluator:

    return FinancialLLMEvaluator(model, tokenizer, **kwargs)
