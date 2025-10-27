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
        if not self._response_sets:
            raise AssertionError("No responses configured")

        collected: list[str] = []
        remaining = n
        while remaining > 0:
            if not self._response_sets:
                raise AssertionError(f"Requested {n} responses but only {len(collected)} available")
            current = self._response_sets[0]
            if not current:
                self._response_sets.pop(0)
                continue

            take = min(remaining, len(current))
            collected.extend(current[:take])
            del current[:take]
            remaining -= take

            if not current:
                self._response_sets.pop(0)

        return [
            {
                "text": text,
                "num_generated_tokens": len(text.split()),
            }
            for text in collected
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


def test_run_escalates_on_weak_consensus():
    entropy_values = [1.5]  # choose mid candidate (n=3)
    response_sets = [
        ["11", "12", "13"],  # no agreement
        ["Answer 14", "14"],  # top-up by 2 samples
    ]
    backend = DummyBackend(entropy_values, response_sets)
    config = make_config(a=1.0, b=2.0, n_candidates=[1, 3, 5], mode="adaptive")
    sampler = AdaptiveSampler(backend, config)
    result = sampler.run("Hard question")

    assert result.n_used == 5
    assert result.votes["14"] == 2
    # No majority reached, but top choice should be the most recent consensus candidate
    assert result.canonical_choice == "14"


def test_fixed_mode_never_topups():
    responses = ["A", "B", "C"]
    backend = DummyBackend(entropy_values=[0.2], response_sets=[responses])
    config = make_config(mode="fixed", n_fixed=3, n_candidates=[3])
    sampler = AdaptiveSampler(backend, config)
    result = sampler.run("Question")

    assert result.n_used == 3
    assert backend._response_sets == []
