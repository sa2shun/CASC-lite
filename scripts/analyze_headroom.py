#!/usr/bin/env python
"""Analyze accuracy plateau and entropy correlations from post-hoc n sweep."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.005  # 0.5 percentage point
DEFAULT_PERSIST = 3


def parse_entropy_sequences(series: pd.Series, k: int = 32) -> np.ndarray:
    def to_avg(raw: str | list) -> float:
        if isinstance(raw, list):
            seq = raw
        else:
            seq = ast.literal_eval(raw) if isinstance(raw, str) and raw else []
        seq = seq[:k]
        if not seq:
            return 0.0
        return float(np.mean(seq))

    return series.apply(to_avg).to_numpy()


def find_plateau(acc: pd.Series, threshold: float, persist: int) -> dict[str, float | int | None]:
    gains = acc.diff().fillna(acc.iloc[0])
    consecutive = 0
    plateau_n = None
    for n, delta in zip(acc.index[1:], gains.iloc[1:]):
        if delta < threshold:
            consecutive += 1
            if consecutive >= persist:
                plateau_n = n
                break
        else:
            consecutive = 0

    return {
        "plateau_n": plateau_n,
        "threshold": threshold,
        "persist": persist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze accuracy vs n plateau and entropy correlations")
    parser.add_argument("--examples_csv", type=Path, required=True, help="CSV with posthoc n columns")
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Plateau gain threshold (in proportion)")
    parser.add_argument("--persist", type=int, default=DEFAULT_PERSIST, help="Number of consecutive gains below threshold")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write summary CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.examples_csv)

    # discover available n columns
    n_columns: List[int] = []
    for col in df.columns:
        if col.startswith("correct_strict_n"):
            try:
                n = int(col[len("correct_strict_n"):])
                n_columns.append(n)
            except ValueError:
                continue
    if not n_columns:
        raise RuntimeError("No posthoc n columns found in CSV")

    n_columns = sorted(n_columns)
    acc = pd.Series({n: df[f"correct_strict_n{n}"].mean() for n in n_columns})
    gains = acc.diff().fillna(acc.iloc[0])

    plateau = find_plateau(acc, args.threshold, args.persist)

    print("Accuracy by n:")
    print(acc.apply(lambda v: f"{v*100:.2f}%"))
    print("\nIncremental gains vs previous:")
    print(gains.apply(lambda v: f"{v*100:.2f} pp"))
    print("\nPlateau detection:")
    print(plateau)

    # best n per example (first success)
    def first_success(row: pd.Series) -> int | None:
        for n in n_columns:
            if row[f"correct_strict_n{n}"]:
                return n
        return None

    best_n = df.apply(first_success, axis=1)
    df["best_n"] = best_n

    entropy_avg = parse_entropy_sequences(df['entropy_tokens'], k=args.max_tokens)
    df['entropy_avg'] = entropy_avg
    df['entropy'] = df.get('entropy', df['entropy_avg'])

    numeric_best = df['best_n'].dropna()
    if not numeric_best.empty:
        corr = df.loc[numeric_best.index, 'entropy'].corr(numeric_best.astype(float))
        print(f"\nCorrelation(entropy, best_n) = {corr:.3f}")

        # headroom: minimal n minus 1
        headroom = numeric_best.astype(float) - 1.0
        df.loc[numeric_best.index, 'headroom'] = headroom
        corr_head = df.loc[numeric_best.index, 'entropy'].corr(headroom)
        print(f"Correlation(entropy, headroom) = {corr_head:.3f}")
    else:
        print("\nNo successful cases to compute best_n correlations.")

    if args.output:
        summary = pd.DataFrame(
            {
                "n": n_columns,
                "accuracy": acc.values,
                "gain": gains.values,
            }
        )
        summary.to_csv(args.output, index=False)
        print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()
