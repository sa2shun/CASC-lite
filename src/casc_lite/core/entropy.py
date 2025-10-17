"""Entropy-related utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class EntropyStats:
    """Container for per-token and average entropy values."""

    average_entropy: float
    token_entropies: list[float]


def compute_entropy_metrics(logits: Sequence[torch.Tensor], K: int) -> EntropyStats:
    """Compute entropy statistics for the first ``K`` tokens.

    Args:
        logits: Sequence of score tensors returned by ``generate`` (time-major).
        K: Number of leading tokens to include in the average.

    Returns:
        ``EntropyStats`` containing the average entropy and per-token values.
    """

    if not logits:
        return EntropyStats(average_entropy=0.0, token_entropies=[])

    token_entropies: list[float] = []
    window = min(len(logits), max(K, 1))

    for step in range(window):
        step_logits = logits[step].detach()
        # Convert to float32 for numerical stability regardless of model dtype.
        step_logits = step_logits.to(torch.float32)
        log_probs = F.log_softmax(step_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_mean = entropy.mean()
        if not torch.isfinite(entropy_mean):
            continue
        token_entropies.append(float(entropy_mean.item()))

    if not token_entropies:
        return EntropyStats(average_entropy=0.0, token_entropies=[])

    average_entropy = float(sum(token_entropies) / len(token_entropies))
    return EntropyStats(average_entropy=average_entropy, token_entropies=token_entropies)
