import dataclasses

import pytest

from casc_lite.backends.base import BaseBackend
from casc_lite.core.entropy import EntropyStats
from casc_lite.core.sampling import AdaptiveSampler
from casc_lite.core.utils import ExperimentConfig


class DummyBackend(BaseBackend):
    def __init__(self, entropy_values, response_sets):
        self._entropy_values = list(entropy_values)
        self._response_sets = [list(responses) for responses in response_sets]

    def prefix_entropy(self, *args, **kwargs):
        value = self._entropy_values.pop(0)
        return EntropyStats(average_entropy=value, token_entropies=[value])

    def generate_n(self, prompt, n, *args, **kwargs):
        responses = self._response_sets.pop(0)
        if len(responses) != n:
            raise AssertionError(f"Expected {n} responses, got {len(responses)}")
        return [
            {
                "text": text,
                "num_generated_tokens": len(text.split()),
            }
            for text in responses
        ]

    def tokenize(self, text):
        return [ord(ch) for ch in text]

    def detokenize(self, token_ids):
        return "".join(chr(x) for x in token_ids)


def make_config(**overrides):
    base = ExperimentConfig()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_decide_n_adaptive_thresholds():
    backend = DummyBackend(entropy_values=[0.5], response_sets=[["a"]])
    config = make_config(a=1.0, b=2.0, n_candidates=[1, 3, 5], mode="adaptive")
    sampler = AdaptiveSampler(backend, config)
    assert sampler.decide_n(0.4) == 1
    assert sampler.decide_n(1.5) == 3
    assert sampler.decide_n(3.0) == 5


def test_run_majority_vote_numeric_priority():
    responses = [
        "The final answer is 11.",
        "11 is correct",
        "Answer: 11",
        "I think it's twelve",
        "Probably eleven",
    ]
    entropy_values = [2.5]
    backend = DummyBackend(entropy_values, [responses])
    config = make_config(a=1.0, b=2.0, n_candidates=[1, 3, 5], mode="adaptive")
    sampler = AdaptiveSampler(backend, config)
    result = sampler.run("What is 5+6?")

    assert result.n_used == 5
    assert result.canonical_choice == "11"
    assert result.votes["11"] == 3
    assert result.chosen.startswith("The final answer is 11")
    expected_avg_tokens = sum(len(r.split()) for r in responses) / len(responses)
    assert result.generated_tokens == pytest.approx(expected_avg_tokens)
