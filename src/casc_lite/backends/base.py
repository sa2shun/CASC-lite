"""Backend abstraction for CASC-lite experiments."""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Sequence

from casc_lite.core.entropy import EntropyStats


class BackendError(RuntimeError):
    """Raised when a backend fails irrecoverably."""


class BaseBackend(abc.ABC):
    """Abstract interface for model backends."""

    @abc.abstractmethod
    def prefix_entropy(
        self,
        prompt: str,
        K: int,
        temperature: float,
        top_p: float | None = None,
        top_k: int | None = None,
        **generate_kwargs: Any,
    ) -> EntropyStats:
        """Estimate the entropy statistics across the first ``K`` tokens."""

    @abc.abstractmethod
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
        """Generate ``n`` samples for the given prompt."""

    @abc.abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Return token ids for the given text."""

    @abc.abstractmethod
    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Convert token ids back to text."""
