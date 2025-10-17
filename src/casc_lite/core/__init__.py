"""Core utilities for CASC-lite."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .entropy import EntropyStats, compute_entropy_metrics
from .evaluate import EvaluationMetrics, Evaluator
from .utils import ExperimentConfig, configure_logging, load_config

if TYPE_CHECKING:  # pragma: no cover - avoids circular imports at runtime
    from .sampling import AdaptiveSampler, SamplingResult

__all__ = [
    "EntropyStats",
    "compute_entropy_metrics",
    "EvaluationMetrics",
    "Evaluator",
    "ExperimentConfig",
    "configure_logging",
    "load_config",
    "AdaptiveSampler",
    "SamplingResult",
]


def __getattr__(name: str):
    if name in {"AdaptiveSampler", "SamplingResult"}:
        from .sampling import AdaptiveSampler, SamplingResult

        return {"AdaptiveSampler": AdaptiveSampler, "SamplingResult": SamplingResult}[name]
    raise AttributeError(f"module 'casc_lite.core' has no attribute {name!r}")
