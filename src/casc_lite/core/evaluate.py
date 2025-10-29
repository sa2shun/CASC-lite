"""Evaluation utilities for CASC-lite."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .utils import extract_number, normalize_text


@dataclass
class EvaluationMetrics:
    accuracy_strict: float
    accuracy_normalized: float
    avg_latency: float
    avg_tokens: float
    efficiency: float


@dataclass
class ExampleResult:
    question: str
    predicted: str
    canonical_predicted: str
    gold: str
    evaluation_mode: str
    correct_strict: bool
    correct_normalized: bool
    latency_seconds: float
    generated_tokens: float
    n_used: int
    entropy: float
    vote_margin: float | None = None
    second_vote_ratio: float | None = None
    token_length_std: float | None = None
    avg_logprob: float | None = None
    logprob_std: float | None = None


class Evaluator:
    """Tracks per-example results and aggregates metrics."""

    def __init__(self) -> None:
        self.examples: List[ExampleResult] = []

    def add_result(
        self,
        *,
        question: str,
        predicted: str,
        canonical_predicted: str,
        gold: str,
        latency_seconds: float,
        generated_tokens: float,
        n_used: int,
        entropy: float,
        vote_margin: float | None = None,
        second_vote_ratio: float | None = None,
        token_length_std: float | None = None,
        avg_logprob: float | None = None,
        logprob_std: float | None = None,
    ) -> None:
        gold_number = extract_number(gold)
        pred_number = extract_number(canonical_predicted)
        if gold_number is not None:
            evaluation_mode = "numeric"
            correct = pred_number == gold_number
            correct_strict = correct
            correct_normalized = correct
        else:
            evaluation_mode = "string"
            correct_strict = predicted.strip() == gold.strip()
            correct_normalized = canonical_predicted == normalize_text(gold)
        example = ExampleResult(
            question=question,
            predicted=predicted,
            canonical_predicted=canonical_predicted,
            gold=gold,
            evaluation_mode=evaluation_mode,
            correct_strict=correct_strict,
            correct_normalized=correct_normalized,
            latency_seconds=latency_seconds,
            generated_tokens=generated_tokens,
            n_used=n_used,
            entropy=entropy,
            vote_margin=vote_margin,
            second_vote_ratio=second_vote_ratio,
            token_length_std=token_length_std,
            avg_logprob=avg_logprob,
            logprob_std=logprob_std,
        )
        self.examples.append(example)

    def aggregate(self) -> EvaluationMetrics:
        if not self.examples:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        total = len(self.examples)
        accuracy_strict = sum(example.correct_strict for example in self.examples) / total
        accuracy_normalized = sum(example.correct_normalized for example in self.examples) / total
        avg_latency = sum(example.latency_seconds for example in self.examples) / total
        avg_tokens = sum(example.generated_tokens for example in self.examples) / total
        efficiency = (
            accuracy_normalized / avg_latency if avg_latency > 0 else 0.0
        )
        return EvaluationMetrics(
            accuracy_strict=accuracy_strict,
            accuracy_normalized=accuracy_normalized,
            avg_latency=avg_latency,
            avg_tokens=avg_tokens,
            efficiency=efficiency,
        )

    def to_csv_rows(self) -> Iterable[Dict[str, str | float | int]]:
        for idx, example in enumerate(self.examples):
            yield {
                "index": idx,
                "question": example.question,
                "predicted": example.predicted,
                "canonical_predicted": example.canonical_predicted,
                "gold": example.gold,
                "evaluation_mode": example.evaluation_mode,
                "correct_strict": int(example.correct_strict),
                "correct_normalized": int(example.correct_normalized),
                "latency_seconds": example.latency_seconds,
                "generated_tokens": example.generated_tokens,
                "n_used": example.n_used,
                "entropy": example.entropy,
                "vote_margin": example.vote_margin,
                "second_vote_ratio": example.second_vote_ratio,
                "token_length_std": example.token_length_std,
                "avg_logprob": example.avg_logprob,
                "logprob_std": example.logprob_std,
            }
