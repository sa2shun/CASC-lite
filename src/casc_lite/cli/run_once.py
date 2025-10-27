"""CLI entry point for running a single CASC-lite experiment."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from casc_lite.backends.base import BaseBackend
from casc_lite.core.evaluate import Evaluator
from casc_lite.core.sampling import AdaptiveSampler
from casc_lite.core.utils import (
    ExperimentConfig,
    configure_logging,
    ensure_output_dir,
    load_config,
    read_jsonl,
    seed_all,
)

logger = logging.getLogger(__name__)


def render_prompt(question: str, template: str | None) -> str:
    """Apply a prompt template to a GSM8K question."""

    clean_question = question.strip()
    if not template:
        return clean_question
    placeholder = "{question}"
    if placeholder in template:
        return template.replace(placeholder, clean_question)
    return f"{template.rstrip()}\n\n{clean_question}"


def summarize_prefix_result(
    *,
    responses: list[str],
    canonical_answers: list[str],
    tokens: list[int],
    sample_latencies: list[float],
    n: int,
) -> Dict[str, Any]:
    """Return majority-vote summary for the first ``n`` responses."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not responses:
        raise ValueError("No responses available to summarise")

    limit = min(n, len(responses))
    subset_responses = responses[:limit]
    subset_canonical = canonical_answers[:limit]
    subset_tokens = tokens[:limit]
    subset_latencies = sample_latencies[:limit] if sample_latencies else []

    first_occurrence: Dict[str, int] = {}
    for idx, canonical in enumerate(subset_canonical):
        first_occurrence.setdefault(canonical, idx)

    counter = Counter(subset_canonical)
    canonical_choice, top_votes = max(
        counter.items(),
        key=lambda item: (item[1], -first_occurrence[item[0]]),
    )
    chosen_idx = first_occurrence[canonical_choice]
    predicted = subset_responses[chosen_idx]

    avg_tokens = float(sum(subset_tokens) / max(limit, 1)) if subset_tokens else 0.0
    latency = float(sum(subset_latencies)) if subset_latencies else 0.0

    return {
        "predicted": predicted,
        "canonical_predicted": canonical_choice,
        "votes": dict(counter),
        "n_used": limit,
        "avg_tokens": avg_tokens,
        "latency": latency,
        "top_votes": top_votes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a CASC-lite experiment once.")
    parser.add_argument("--config", default="src/casc_lite/config/default.yaml", help="YAML config path")
    parser.add_argument("--model", help="Model name or path", default=None)
    parser.add_argument("--backend", choices=["hf_transformers", "vllm"], default=None)
    parser.add_argument("--data", help="Path to GSM8K-style JSONL data", default=None)
    parser.add_argument("--mode", choices=["adaptive", "fixed"], default=None)
    parser.add_argument("--n_fixed", type=int, default=None, help="Number of samples if mode=fixed")
    parser.add_argument("--K", type=int, default=None, help="Number of prefix tokens for entropy")
    parser.add_argument("--a", type=float, default=None, help="Lower entropy threshold")
    parser.add_argument("--b", type=float, default=None, help="Upper entropy threshold")
    parser.add_argument("--T", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--n_candidates", default=None, help="Comma-separated candidates list")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--log_level", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retry_on_error", type=int, default=None)
    parser.add_argument("--entropy_window", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--posthoc_n_values",
        default=None,
        help="Comma-separated list of sample counts to evaluate post-hoc",
    )
    parser.add_argument(
        "--save_entropy_tokens",
        type=int,
        default=None,
        help="Number of prefix entropy tokens to persist per example (0 disables)",
    )
    parser.add_argument(
        "--save_completions",
        action="store_true",
        help="Dump raw completions to JSON for inspection",
    )
    parser.add_argument(
        "--dump_completions",
        action="store_true",
        dest="save_completions",
        help="Alias for --save_completions",
    )
    return parser


def create_backend(config: ExperimentConfig) -> BaseBackend:
    if config.backend == "hf_transformers":
        from casc_lite.backends.hf_transformers import HFTransformersBackend

        return HFTransformersBackend(model_name=config.model, device=config.device)
    if config.backend == "vllm":
        from casc_lite.backends.vllm import VLLMBackend

        return VLLMBackend()  # Raises NotImplementedError per specification
    raise ValueError(f"Unsupported backend: {config.backend}")


def run_experiment(
    config: ExperimentConfig,
    *,
    backend: Optional[BaseBackend] = None,
    save_completions: bool = False,
) -> Dict[str, Any]:
    seed_all(config.seed)
    data_records = read_jsonl(config.data)
    if not data_records:
        raise ValueError(f"Dataset {config.data} is empty")

    backend = backend or create_backend(config)
    sampler = AdaptiveSampler(backend=backend, config=config)
    evaluator = Evaluator()

    posthoc_ns = sorted({int(n) for n in config.posthoc_n_values if int(n) > 0})
    posthoc_evaluators: Dict[int, Evaluator] = {n: Evaluator() for n in posthoc_ns}

    n_values: list[int] = []
    entropies: list[float] = []
    entropy_tokens_history: list[list[float]] = []
    completions_dump: list[Dict[str, Any]] = []

    total_examples = len(data_records)

    for idx, example in enumerate(data_records):
        question = str(example["question"]).strip()
        gold = str(example.get("answer", "")).strip()
        prompt = render_prompt(question, config.prompt_template)
        result = sampler.run(prompt)

        entropy_value = result.entropy.average_entropy
        entropies.append(entropy_value)
        entropy_tokens = result.entropy.token_entropies
        if config.save_entropy_tokens and config.save_entropy_tokens > 0:
            entropy_tokens = entropy_tokens[: config.save_entropy_tokens]
        entropy_tokens_history.append(entropy_tokens)

        evaluator.add_result(
            question=question,
            predicted=result.chosen,
            canonical_predicted=result.canonical_choice,
            gold=gold,
            latency_seconds=result.latency_seconds,
            generated_tokens=result.generated_tokens,
            n_used=result.n_used,
            entropy=entropy_value,
        )
        n_values.append(result.n_used)

        prefix_summaries: Dict[int, Dict[str, Any]] = {}
        for n in posthoc_ns:
            summary_n = summarize_prefix_result(
                responses=result.responses,
                canonical_answers=result.canonical_answers,
                tokens=result.tokens,
                sample_latencies=result.sample_latencies,
                n=n,
            )
            prefix_summaries[n] = summary_n
            posthoc_evaluators[n].add_result(
                question=question,
                predicted=summary_n["predicted"],
                canonical_predicted=summary_n["canonical_predicted"],
                gold=gold,
                latency_seconds=summary_n["latency"],
                generated_tokens=summary_n["avg_tokens"],
                n_used=summary_n["n_used"],
                entropy=entropy_value,
            )

        if save_completions:
            completions_dump.append(
                {
                    "question": question,
                    "responses": result.responses,
                    "canonical_answers": result.canonical_answers,
                    "votes": result.votes,
                    "chosen": result.chosen,
                    "canonical_choice": result.canonical_choice,
                    "entropy": entropy_value,
                    "entropy_tokens": entropy_tokens,
                    "latency": result.latency_seconds,
                    "generated_tokens": result.generated_tokens,
                    "n_used": result.n_used,
                    "tokens": result.tokens,
                    "sample_latencies": result.sample_latencies,
                    "posthoc": prefix_summaries if posthoc_ns else None,
                    "gold": gold,
                }
            )

        logger.info(
            "Progress: %d/%d (%.1f%%)",
            idx + 1,
            total_examples,
            (idx + 1) * 100.0 / total_examples,
        )

    metrics = evaluator.aggregate()
    avg_n = sum(n_values) / len(n_values)
    avg_entropy = sum(entropies) / len(entropies)
    correct_strict = sum(1 for example in evaluator.examples if example.correct_strict)
    correct_normalized = sum(1 for example in evaluator.examples if example.correct_normalized)

    posthoc_summary: Dict[int, Dict[str, Any]] = {}
    for n, posthoc_eval in posthoc_evaluators.items():
        posthoc_metrics = posthoc_eval.aggregate()
        total = len(posthoc_eval.examples)
        if total == 0:
            continue
        posthoc_summary[n] = {
            "metrics": posthoc_metrics,
            "avg_n": sum(example.n_used for example in posthoc_eval.examples) / total,
            "dataset_size": total,
            "correct_strict": sum(example.correct_strict for example in posthoc_eval.examples),
            "correct_normalized": sum(example.correct_normalized for example in posthoc_eval.examples),
        }

    return {
        "metrics": metrics,
        "evaluator": evaluator,
        "avg_n": avg_n,
        "avg_entropy": avg_entropy,
        "dataset_size": len(data_records),
        "completions": completions_dump,
        "correct_strict_count": correct_strict,
        "correct_normalized_count": correct_normalized,
        "posthoc_evaluators": posthoc_evaluators,
        "posthoc_summary": posthoc_summary,
        "posthoc_n_values": posthoc_ns,
        "entropy_tokens_history": entropy_tokens_history,
    }


def persist_results(
    config: ExperimentConfig,
    summary: Dict[str, Any],
    output_dir: Path,
    run_id: str,
    *,
    save_completions: bool,
) -> Dict[str, Path | None]:
    metrics = summary["metrics"]
    evaluator: Evaluator = summary["evaluator"]
    posthoc_evaluators: Dict[int, Evaluator] = summary.get("posthoc_evaluators", {})
    posthoc_ns = sorted(posthoc_evaluators.keys())
    entropy_tokens_history = summary.get("entropy_tokens_history") or []

    per_example_path = output_dir / f"{run_id}_examples.csv"
    with open(per_example_path, "w", newline="", encoding="utf-8") as fp:
        base_fieldnames = [
            "index",
            "question",
            "predicted",
            "canonical_predicted",
            "gold",
            "evaluation_mode",
            "correct_strict",
            "correct_normalized",
            "latency_seconds",
            "generated_tokens",
            "n_used",
            "entropy",
            "entropy_tokens",
        ]

        scenario_fieldnames: list[str] = []
        for n in posthoc_ns:
            suffix = f"n{n}"
            scenario_fieldnames.extend(
                [
                    f"predicted_{suffix}",
                    f"canonical_predicted_{suffix}",
                    f"correct_strict_{suffix}",
                    f"correct_normalized_{suffix}",
                    f"latency_seconds_{suffix}",
                    f"generated_tokens_{suffix}",
                    f"n_used_{suffix}",
                ]
            )

        fieldnames = base_fieldnames + scenario_fieldnames
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        primary_examples = evaluator.examples
        scenario_examples = {n: posthoc_evaluators[n].examples for n in posthoc_ns}

        for idx, example in enumerate(primary_examples):
            row = {
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
                "entropy_tokens": json.dumps(
                    entropy_tokens_history[idx] if idx < len(entropy_tokens_history) else []
                ),
            }

            for n in posthoc_ns:
                scenario = scenario_examples[n][idx]
                suffix = f"n{n}"
                row[f"predicted_{suffix}"] = scenario.predicted
                row[f"canonical_predicted_{suffix}"] = scenario.canonical_predicted
                row[f"correct_strict_{suffix}"] = int(scenario.correct_strict)
                row[f"correct_normalized_{suffix}"] = int(scenario.correct_normalized)
                row[f"latency_seconds_{suffix}"] = scenario.latency_seconds
                row[f"generated_tokens_{suffix}"] = scenario.generated_tokens
                row[f"n_used_{suffix}"] = scenario.n_used

            writer.writerow(row)

    aggregate_path = output_dir / "aggregate.csv"
    summary_row = {
        "run_id": run_id,
        "mode": config.mode,
        "model": config.model,
        "backend": config.backend,
        "n_fixed": config.n_fixed or "",
        "K": config.K,
        "a": config.a,
        "b": config.b,
        "temperature": config.T,
        "top_p": config.top_p if config.top_p is not None else "",
        "top_k": config.top_k if config.top_k is not None else "",
        "dataset_size": summary["dataset_size"],
        "accuracy_strict": metrics.accuracy_strict,
        "accuracy_normalized": metrics.accuracy_normalized,
        "avg_latency": metrics.avg_latency,
        "avg_tokens": metrics.avg_tokens,
        "efficiency": metrics.efficiency,
        "avg_n": summary["avg_n"],
        "avg_entropy": summary["avg_entropy"],
    }

    for n in sorted(summary.get("posthoc_summary", {}).keys()):
        info = summary["posthoc_summary"][n]
        metrics_n = info["metrics"]
        suffix = f"n{n}"
        summary_row[f"accuracy_strict_{suffix}"] = metrics_n.accuracy_strict
        summary_row[f"accuracy_normalized_{suffix}"] = metrics_n.accuracy_normalized
        summary_row[f"avg_latency_{suffix}"] = metrics_n.avg_latency
        summary_row[f"avg_tokens_{suffix}"] = metrics_n.avg_tokens
        summary_row[f"efficiency_{suffix}"] = metrics_n.efficiency
        summary_row[f"avg_n_{suffix}"] = info["avg_n"]
        summary_row[f"correct_strict_count_{suffix}"] = info["correct_strict"]
        summary_row[f"correct_normalized_count_{suffix}"] = info["correct_normalized"]

    write_header = not aggregate_path.exists()
    with open(aggregate_path, "a", newline="", encoding="utf-8") as fp:
        fieldnames = list(summary_row.keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    completions_path: Path | None = None
    if save_completions and summary["completions"]:
        completions_path = output_dir / f"{run_id}_completions.json"
        with open(completions_path, "w", encoding="utf-8") as fp:
            json.dump(summary["completions"], fp, ensure_ascii=False, indent=2)

    return {
        "per_example": per_example_path,
        "aggregate": aggregate_path,
        "completions": completions_path,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "save_completions"} and v is not None}
    config = load_config(args.config, overrides)

    configure_logging(config.log_level)
    output_dir = ensure_output_dir(config.output_dir)

    logger.info("Loaded config: %s", config)

    summary = run_experiment(config, save_completions=args.save_completions)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{config.mode}"
    paths = persist_results(
        config,
        summary,
        output_dir,
        run_id,
        save_completions=args.save_completions,
    )

    metrics = summary["metrics"]
    dataset_size = summary["dataset_size"]
    strict_hits = summary["correct_strict_count"]
    normalized_hits = summary["correct_normalized_count"]
    logger.info(
        "Run %s complete: strict=%.3f (%d/%d) normalized=%.3f (%d/%d) latency=%.3fs tokens=%.2f efficiency=%.3f",
        run_id,
        metrics.accuracy_strict,
        strict_hits,
        dataset_size,
        metrics.accuracy_normalized,
        normalized_hits,
        dataset_size,
        metrics.avg_latency,
        metrics.avg_tokens,
        metrics.efficiency,
    )
    logger.info("Per-example results: %s", paths["per_example"])
    logger.info("Aggregate updated: %s", paths["aggregate"])
    if paths["completions"]:
        logger.info("Completions saved to: %s", paths["completions"])


if __name__ == "__main__":
    main()
