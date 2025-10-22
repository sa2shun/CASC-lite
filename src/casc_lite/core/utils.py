"""Utility helpers for CASC-lite."""
from __future__ import annotations

import dataclasses
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy optional for seed fallback
    np = None
import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    backend: str = "hf_transformers"
    seed: int = 42
    K: int = 8
    a: float = 1.3
    b: float = 2.0
    T: float = 0.7
    beta: float = 1.0
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    n_candidates: List[int] = field(default_factory=lambda: [1, 3, 5])
    mode: str = "adaptive"
    n_fixed: Optional[int] = None
    data: str = "data/gsm8k_mini.jsonl"
    output_dir: str = "results"
    entropy_window: int = 32
    retry_on_error: int = 1
    log_level: str = "INFO"
    device: Optional[str] = None
    prompt_template: Optional[str] = (
        "You are a helpful math tutor. Solve the following problem step by step "
        "and provide only the final numeric answer.\n\nQuestion: {question}\nAnswer:"
    )


def _coerce_optional(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return value


def load_config(path: str | Path | None, overrides: Dict[str, Any] | None = None) -> ExperimentConfig:
    """Load experiment configuration from YAML and apply overrides."""

    base = dataclasses.asdict(ExperimentConfig())

    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must define a mapping")
        base.update(loaded)

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            base[key] = value

    # Normalise list-type parameters.
    if isinstance(base.get("n_candidates"), str):
        base["n_candidates"] = [int(x) for x in base["n_candidates"].split(",") if x]
    elif isinstance(base.get("n_candidates"), Iterable) and not isinstance(base.get("n_candidates"), list):
        base["n_candidates"] = [int(x) for x in base["n_candidates"]]
    else:
        base["n_candidates"] = [int(x) for x in base.get("n_candidates", [])]

    if not base["n_candidates"]:
        base["n_candidates"] = [1]

    base["mode"] = str(base.get("mode", "adaptive")).lower()
    base["log_level"] = str(base.get("log_level", "INFO")).upper()
    base["top_p"] = _coerce_optional(base.get("top_p"))
    if base.get("top_p") is not None:
        base["top_p"] = float(base["top_p"])
    base["top_k"] = _coerce_optional(base.get("top_k"))
    if base.get("top_k") is not None:
        base["top_k"] = int(base["top_k"])

    int_fields = ["seed", "K", "max_new_tokens", "min_new_tokens", "retry_on_error", "entropy_window"]
    float_fields = ["a", "b", "T", "beta"]

    for field_name in int_fields:
        if field_name in base and base[field_name] is not None:
            base[field_name] = int(base[field_name])

    for field_name in float_fields:
        if field_name in base and base[field_name] is not None:
            base[field_name] = float(base[field_name])

    if base.get("n_fixed") is not None:
        base["n_fixed"] = int(base["n_fixed"])

    if base.get("prompt_template") is not None:
        base["prompt_template"] = str(base["prompt_template"])

    config = ExperimentConfig(**base)
    return config


def seed_all(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_number(text: str) -> Optional[str]:
    if not text:
        return None
    match = None
    for candidate in reversed(list(NUMBER_RE.finditer(text))):
        match = candidate
        break
    if match is None:
        return None
    integer, decimal = match.group("int"), match.group("dec")
    if decimal:
        value = f"{integer}.{decimal}"
    else:
        value = integer
    return value.lstrip("+")


def ensure_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


NUMBER_RE = re.compile(r"(?P<int>[+-]?\d+)(?:\.(?P<dec>\d+))?", re.MULTILINE)
