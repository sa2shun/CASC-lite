import pytest
import torch

from casc_lite.core.entropy import EntropyStats, compute_entropy_metrics


def test_compute_entropy_uniform_distribution():
    logits = [torch.zeros((1, 2)) for _ in range(2)]
    stats = compute_entropy_metrics(logits, K=2)
    assert isinstance(stats, EntropyStats)
    assert stats.average_entropy == pytest.approx(torch.log(torch.tensor(2.0)).item(), rel=1e-3)
    assert len(stats.token_entropies) == 2


def test_compute_entropy_partial_window():
    logits = [torch.tensor([[4.0, 0.0, 0.0]])]
    stats = compute_entropy_metrics(logits, K=5)
    assert stats.average_entropy < 0.2
    assert stats.token_entropies[0] == pytest.approx(stats.average_entropy)


def test_compute_entropy_handles_non_finite():
    logits = [torch.tensor([[float("nan"), 0.0]])]
    stats = compute_entropy_metrics(logits, K=2)
    assert stats.average_entropy == 0.0
    assert stats.token_entropies == []
