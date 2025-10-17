"""CLI for exporting plots from CSV results."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export accuracy-latency plots from aggregate CSV.")
    parser.add_argument("--csv", required=True, help="Path to aggregate CSV file")
    parser.add_argument("--output_dir", default=None, help="Directory to save plots (defaults to CSV parent)")
    parser.add_argument("--scatter_name", default="accuracy_latency_scatter.png")
    parser.add_argument("--pareto_name", default="accuracy_latency_pareto.png")
    return parser


def compute_pareto(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values("avg_latency")
    pareto_rows = []
    best_accuracy = -1.0
    for _, row in df_sorted.iterrows():
        acc = row["accuracy_normalized"]
        if acc >= best_accuracy:
            pareto_rows.append(row)
            best_accuracy = acc
    return pd.DataFrame(pareto_rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV file is empty")

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["avg_latency"], df["accuracy_normalized"], c=df["avg_n"], cmap="viridis", s=80)
    for _, row in df.iterrows():
        plt.annotate(row["mode"], (row["avg_latency"], row["accuracy_normalized"]), fontsize=8, alpha=0.7)
    plt.colorbar(scatter, label="Average n")
    plt.xlabel("Average Latency (s)")
    plt.ylabel("Normalized Accuracy")
    plt.title("Accuracy vs Latency")
    scatter_path = output_dir / args.scatter_name
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    pareto_df = compute_pareto(df)
    plt.figure(figsize=(8, 6))
    plt.plot(df["avg_latency"], df["accuracy_normalized"], "o", alpha=0.3, label="All runs")
    if not pareto_df.empty:
        plt.plot(
            pareto_df["avg_latency"],
            pareto_df["accuracy_normalized"],
            "-o",
            color="tab:red",
            label="Pareto front",
        )
    plt.xlabel("Average Latency (s)")
    plt.ylabel("Normalized Accuracy")
    plt.title("Pareto Front: Accuracy vs Latency")
    plt.legend()
    pareto_path = output_dir / args.pareto_name
    plt.tight_layout()
    plt.savefig(pareto_path, dpi=300)
    plt.close()

    logger.info("Saved scatter plot to %s", scatter_path)
    logger.info("Saved Pareto plot to %s", pareto_path)


if __name__ == "__main__":
    main()
