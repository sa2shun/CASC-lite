import pytest

from casc_lite.core.evaluate import Evaluator


def test_numeric_evaluation_counts_as_correct():
    evaluator = Evaluator()
    evaluator.add_result(
        question="2+3?",
        predicted="The answer is 5.",
        canonical_predicted="5",
        gold="5",
        latency_seconds=0.5,
        generated_tokens=10,
        n_used=3,
        entropy=1.2,
    )
    metrics = evaluator.aggregate()
    assert metrics.accuracy_strict == pytest.approx(1.0)
    assert metrics.accuracy_normalized == pytest.approx(1.0)


def test_string_evaluation_strict_vs_normalized():
    evaluator = Evaluator()
    evaluator.add_result(
        question="Capital of France",
        predicted="Paris",
        canonical_predicted="paris",
        gold="PARIS",
        latency_seconds=0.2,
        generated_tokens=5,
        n_used=1,
        entropy=0.8,
    )
    evaluator.add_result(
        question="Capital of Spain",
        predicted="Barca",
        canonical_predicted="barca",
        gold="Madrid",
        latency_seconds=0.3,
        generated_tokens=6,
        n_used=1,
        entropy=1.1,
    )
    metrics = evaluator.aggregate()
    assert metrics.accuracy_strict == pytest.approx(0.0)
    assert metrics.accuracy_normalized == pytest.approx(0.5)
    rows = list(evaluator.to_csv_rows())
    assert rows[0]["correct_strict"] == 0
    assert rows[0]["correct_normalized"] == 1
