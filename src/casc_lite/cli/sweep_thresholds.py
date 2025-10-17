"""CLI to sweep entropy thresholds."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
from typing import List

from casc_lite.cli.run_once import create_backend, persist_results, run_experiment
from casc_lite.core.utils import (
    ExperimentConfig,
    configure_logging,
    ensure_output_dir,
    load_config,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep entropy thresholds (a, b) for CASC-lite.")
    parser.add_argument("--config", default="src/casc_lite/config/default.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--backend", choices=["hf_transformers", "vllm"], default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--a_list", required=True, help="Comma-separated list of 'a' values")
    parser.add_argument("--b_list", required=True, help="Comma-separated list of 'b' values")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_level", default=None)
    parser.add_argument("--save_completions", action="store_true")
    return parser


def parse_float_list(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "a_list", "b_list", "save_completions"} and value is not None
    }
    config = load_config(args.config, overrides)
    config = dataclasses.replace(config, mode="adaptive")

    configure_logging(config.log_level)
    output_dir = ensure_output_dir(config.output_dir)

    a_values = parse_float_list(args.a_list)
    b_values = parse_float_list(args.b_list)
    if not a_values or not b_values:
        raise ValueError("Both a_list and b_list must be non-empty")

    logger.info("Sweeping a values %s and b values %s", a_values, b_values)

    backend = create_backend(config)

    for a in a_values:
        for b in b_values:
            if a >= b:
                logger.warning("Skipping combination a=%.3f, b=%.3f because a >= b", a, b)
                continue
            sweep_config = dataclasses.replace(config, a=a, b=b)
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_adaptive_a{a:.2f}_b{b:.2f}".replace("/", "-")
            logger.info("Running sweep for a=%.3f, b=%.3f", a, b)
            summary = run_experiment(
                sweep_config,
                backend=backend,
                save_completions=args.save_completions,
            )
            paths = persist_results(
                sweep_config,
                summary,
                output_dir,
                run_id,
                save_completions=args.save_completions,
            )
            metrics = summary["metrics"]
            logger.info(
                "Completed sweep a=%.3f b=%.3f -> strict=%.3f normalized=%.3f latency=%.3fs",
                a,
                b,
                metrics.accuracy_strict,
                metrics.accuracy_normalized,
                metrics.avg_latency,
            )
            if paths["completions"]:
                logger.info("Completions saved to %s", paths["completions"])


if __name__ == "__main__":
    main()
