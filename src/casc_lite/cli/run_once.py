"""CLI entry point for running a single CASC-lite experiment."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
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
    n_values: list[int] = []
    entropies: list[float] = []
    completions_dump: list[Dict[str, Any]] = []

    for example in data_records:
        question = str(example["question"]).strip()
        gold = str(example.get("answer", "")).strip()
        prompt = render_prompt(question, config.prompt_template)
        result = sampler.run(prompt)
        evaluator.add_result(
            question=question,
            predicted=result.chosen,
            canonical_predicted=result.canonical_choice,
            gold=gold,
            latency_seconds=result.latency_seconds,
            generated_tokens=result.generated_tokens,
            n_used=result.n_used,
            entropy=result.entropy.average_entropy,
        )
        n_values.append(result.n_used)
        entropies.append(result.entropy.average_entropy)
        if save_completions:
            completions_dump.append(
                {
                    "question": question,
                    "responses": result.responses,
                    "votes": result.votes,
                    "chosen": result.chosen,
                    "canonical_choice": result.canonical_choice,
                    "entropy": result.entropy.average_entropy,
                    "latency": result.latency_seconds,
                    "generated_tokens": result.generated_tokens,
                    "n_used": result.n_used,
                    "gold": gold,
                }
            )

    metrics = evaluator.aggregate()
    avg_n = sum(n_values) / len(n_values)
    avg_entropy = sum(entropies) / len(entropies)
    correct_strict = sum(1 for example in evaluator.examples if example.correct_strict)
    correct_normalized = sum(1 for example in evaluator.examples if example.correct_normalized)

    return {
        "metrics": metrics,
        "evaluator": evaluator,
        "avg_n": avg_n,
        "avg_entropy": avg_entropy,
        "dataset_size": len(data_records),
        "completions": completions_dump,
        "correct_strict_count": correct_strict,
        "correct_normalized_count": correct_normalized,
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

    per_example_path = output_dir / f"{run_id}_examples.csv"
    with open(per_example_path, "w", newline="", encoding="utf-8") as fp:
        fieldnames = [
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
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluator.to_csv_rows():
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
