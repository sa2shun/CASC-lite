"""Backend interfaces for CASC-lite."""
from __future__ import annotations

from .base import BaseBackend, BackendError

__all__ = ["BaseBackend", "BackendError", "HFTransformersBackend", "VLLMBackend"]

try:
    from .hf_transformers import HFTransformersBackend  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    HFTransformersBackend = None  # type: ignore

try:
    from .vllm import VLLMBackend  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    VLLMBackend = None  # type: ignore
