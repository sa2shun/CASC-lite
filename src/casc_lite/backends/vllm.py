"""vLLM backend placeholder."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from .base import BaseBackend


class VLLMBackend(BaseBackend):
    """Placeholder for future vLLM integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("vLLM backend is not yet implemented")

    def prefix_entropy(
        self,
        prompt: str,
        K: int,
        temperature: float,
        top_p: float | None = None,
        top_k: int | None = None,
        **generate_kwargs: Any,
    ) -> float:
        raise NotImplementedError

    def generate_n(
        self,
        prompt: str,
        n: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        max_new_tokens: int,
        min_new_tokens: int,
        **generate_kwargs: Any,
    ) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

    def tokenize(self, text: str) -> list[int]:
        raise NotImplementedError

    def detokenize(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError
