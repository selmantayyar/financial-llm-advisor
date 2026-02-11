
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import logging
import re
import sys
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:

    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
    append_timestamp: bool = True,
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append_timestamp:
        metrics = {
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logging.getLogger(__name__).info(f"Metrics saved to {output_path}")
    return output_path


def load_metrics(input_path: Union[str, Path]) -> Dict[str, Any]:

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {input_path}")

    with open(input_path, "r") as f:
        metrics = json.load(f)

    logging.getLogger(__name__).info(f"Metrics loaded from {input_path}")
    return metrics


def format_financial_response(
    analysis: str,
    confidence: float,
    entities: Optional[List[str]] = None,
    latency_ms: Optional[float] = None,
) -> str:
    
    lines = []

    lines.append("=" * 60)
    lines.append("FINANCIAL ANALYSIS")
    lines.append("=" * 60)

    lines.append("")
    lines.append(analysis)
    lines.append("")

    lines.append("-" * 60)

    confidence_pct = confidence * 100
    if confidence_pct >= 80:
        confidence_indicator = "HIGH"
    elif confidence_pct >= 60:
        confidence_indicator = "MEDIUM"
    else:
        confidence_indicator = "LOW"

    lines.append(f"Confidence: {confidence_pct:.1f}% ({confidence_indicator})")

    if entities:
        lines.append(f"Entities: {', '.join(entities)}")

    if latency_ms is not None:
        lines.append(f"Latency: {latency_ms:.1f}ms")

    lines.append("=" * 60)

    return "\n".join(lines)

FINANCIAL_ENTITY_PATTERNS = {
    "company": [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.|Corp\.|Corporation|Company|Co\.|Ltd\.|LLC|PLC))\b",
        r"\b(Apple|Microsoft|Google|Amazon|Meta|Tesla|Nvidia|Netflix)\b",
        r"\b([A-Z]{2,5})\b(?=\s+(?:stock|shares|equity))",  # Stock tickers
    ],
    "person": [
        r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s+(?:CEO|CFO|CTO|COO|Chairman|President|Director))",
        r"\b(?:CEO|CFO|CTO|COO|Chairman|President|Director)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
    ],
    "money": [
        r"\$[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|trillion|B|M|T)?",
        r"[\d,]+(?:\.\d{1,2})?\s*(?:billion|million|trillion)\s*(?:dollars|USD)?",
    ],
    "percentage": [
        r"[\d.]+\s*%",
        r"[\d.]+\s*percent",
        r"[\d.]+\s*basis\s*points?",
    ],
    "date": [
        r"\b(?:Q[1-4])\s*(?:20\d{2}|'\d{2})\b",  # Quarters
        r"\b(?:FY|CY)\s*20\d{2}\b",  # Fiscal/Calendar year
        r"\b20\d{2}\b(?=\s+(?:annual|quarterly|earnings))",
    ],
    "regulation": [
        r"\b(?:SEC|FINRA|GAAP|IFRS|Dodd-Frank|SOX|Sarbanes-Oxley)\b",
    ],
}


def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    
    if entity_types is None:
        entity_types = list(FINANCIAL_ENTITY_PATTERNS.keys())

    results: Dict[str, List[str]] = {t: [] for t in entity_types}

    for entity_type in entity_types:
        if entity_type not in FINANCIAL_ENTITY_PATTERNS:
            continue

        patterns = FINANCIAL_ENTITY_PATTERNS[entity_type]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)

            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                if match and match not in results[entity_type]:
                    results[entity_type].append(match.strip())

    return results


def flatten_entities(entities_dict: Dict[str, List[str]]) -> List[str]:
    
    all_entities = []
    for entity_list in entities_dict.values():
        for entity in entity_list:
            if entity not in all_entities:
                all_entities.append(entity)
    return all_entities


def calculate_token_cost(
    num_tokens: int,
    model_name: str = "phi-3.5-mini",
    is_input: bool = True,
) -> float:
    

    COSTS_PER_MILLION = {
        "phi-3.5-mini": {"input": 0.10, "output": 0.15},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
    }

    model_costs = COSTS_PER_MILLION.get(model_name.lower(), COSTS_PER_MILLION["phi-3.5-mini"])

    cost_type = "input" if is_input else "output"
    cost_per_token = model_costs[cost_type] / 1_000_000

    return num_tokens * cost_per_token


def format_duration(seconds: float) -> str:
    
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    secs = seconds % 60

    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"

    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def create_prompt(
    user_query: str,
    system_prompt: str,
    context: Optional[str] = None,
    format_type: str = "phi",
) -> str:
    
    if format_type == "phi":

        prompt = f"<|system|>\n{system_prompt}<|end|>\n"
        if context:
            prompt += f"<|user|>\nContext: {context}\n\nQuestion: {user_query}<|end|>\n"
        else:
            prompt += f"<|user|>\n{user_query}<|end|>\n"
        prompt += "<|assistant|>\n"

    elif format_type == "llama":

        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        if context:
            prompt += f"Context: {context}\n\nQuestion: {user_query} [/INST]"
        else:
            prompt += f"{user_query} [/INST]"

    elif format_type == "chatml":

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        if context:
            prompt += f"<|im_start|>user\nContext: {context}\n\nQuestion: {user_query}<|im_end|>\n"
        else:
            prompt += f"<|im_start|>user\n{user_query}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    else:

        prompt = f"System: {system_prompt}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"User: {user_query}\n\nAssistant: "

    return prompt
