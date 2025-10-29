#!/usr/bin/env python
"""Evaluate cascade gating thresholds using trained entropy-stage models."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MAX_TOKENS = 32


def load_entropy_matrix(df: pd.DataFrame, max_tokens: int) -> np.ndarray:
    def to_sequence(raw):
        if isinstance(raw, list):
            seq = raw
        else:
            seq = ast.literal_eval(raw) if isinstance(raw, str) and raw else []
        seq = seq[:max_tokens]
        if len(seq) < max_tokens:
            seq = seq + [0.0]*(max_tokens-len(seq))
        return seq
    return np.vstack(df['entropy_tokens'].apply(to_sequence).to_numpy())


def load_model(model_dir: Path, label: str, X: np.ndarray) -> np.ndarray:
    with (model_dir / f"cascade_scaler_{label}.json").open() as fp:
        scaler_info = json.load(fp)
    coef_file = np.load(model_dir / f"cascade_logreg_{label}.npz")
    mean = np.array(scaler_info['mean'])
    scale = np.array(scaler_info['scale'])
    coef = coef_file['coef']
    intercept = coef_file['intercept']
    X_scaled = (X - mean) / scale
    logits = X_scaled @ coef.T + intercept
    return 1 / (1 + np.exp(-logits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cascade thresholds on cached data.")
    parser.add_argument(
        "--examples_csv",
        type=Path,
        default=Path("results/suite_20251027_220603/20251029_040055_fixed_examples.csv"),
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("models/entropy_cascade"),
    )
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    df = pd.read_csv(args.examples_csv)
    entropy_matrix = load_entropy_matrix(df, args.max_tokens)

    # Stage probabilities using final models
    p1 = load_model(args.model_dir, "stage1", entropy_matrix).ravel()

    def build_stage2_features() -> np.ndarray:
        match_13 = (df['canonical_predicted_n1'] == df['canonical_predicted_n3']).astype(float).to_numpy().reshape(-1,1)
        return np.hstack([entropy_matrix, p1.reshape(-1,1), match_13])

    X2 = build_stage2_features()
    p3 = load_model(args.model_dir, "stage2", X2).ravel()

    match_13 = (df['canonical_predicted_n1'] == df['canonical_predicted_n3']).astype(float).to_numpy().reshape(-1,1)
    match_35 = (df['canonical_predicted_n3'] == df['canonical_predicted_n5']).astype(float).to_numpy().reshape(-1,1)
    match_15 = (df['canonical_predicted_n1'] == df['canonical_predicted_n5']).astype(float).to_numpy().reshape(-1,1)
    X3 = np.hstack([entropy_matrix, p1.reshape(-1,1), p3.reshape(-1,1), match_13, match_35, match_15])
    p5 = load_model(args.model_dir, "stage3", X3).ravel()

    match_57 = (df['canonical_predicted_n5'] == df['canonical_predicted_n7']).astype(float).to_numpy().reshape(-1,1)
    match_37 = (df['canonical_predicted_n3'] == df['canonical_predicted_n7']).astype(float).to_numpy().reshape(-1,1)
    match_17 = (df['canonical_predicted_n1'] == df['canonical_predicted_n7']).astype(float).to_numpy().reshape(-1,1)
    X4 = np.hstack([
        entropy_matrix,
        p1.reshape(-1,1),
        p3.reshape(-1,1),
        p5.reshape(-1,1),
        match_13,
        match_35,
        match_15,
        match_57,
        match_37,
        match_17,
    ])
    p7 = load_model(args.model_dir, "stage4", X4).ravel()

    correct = {n: df[f"correct_strict_n{n}"].to_numpy() for n in [1,3,5,7]}
    latency = {n: df[f"latency_seconds_n{n}"].to_numpy() for n in [1,3,5,7]}
    tokens = {n: df[f"generated_tokens_n{n}"].to_numpy() for n in [1,3,5,7]}
    N = len(df)

    threshold_grid = np.linspace(0.3, 0.7, 9)
    records = []

    for t1 in threshold_grid:
        for t3 in threshold_grid:
            for t5 in threshold_grid:
                choices = np.empty(N, dtype=int)
                for i in range(N):
                    if p1[i] >= t1:
                        choices[i] = 1
                    else:
                        if p3[i] >= t3:
                            choices[i] = 3
                        else:
                            if p5[i] >= t5:
                                choices[i] = 5
                            else:
                                choices[i] = 7
                acc = sum(correct[n][choices == n].sum() for n in [1,3,5,7]) / N
                avg_n = sum((choices == n).sum() * n for n in [1,3,5,7]) / N
                avg_latency = sum(latency[n][choices == n].sum() for n in [1,3,5,7]) / N
                avg_tokens = sum(tokens[n][choices == n].sum() for n in [1,3,5,7]) / N
                records.append({
                    't1': t1,
                    't3': t3,
                    't5': t5,
                    'accuracy': acc,
                    'avg_n': avg_n,
                    'avg_latency': avg_latency,
                    'avg_tokens': avg_tokens,
                    'share_n1': (choices == 1).mean(),
                    'share_n3': (choices == 3).mean(),
                    'share_n5': (choices == 5).mean(),
                    'share_n7': (choices == 7).mean(),
                })

    df_records = pd.DataFrame(records)
    df_records.to_csv(args.model_dir / 'cascade_threshold_grid.csv', index=False)
    print('Saved grid search to', args.model_dir / 'cascade_threshold_grid.csv')

    for limit in [6,5,4,3.5]:
        subset = df_records[df_records['avg_n'] <= limit]
        if subset.empty:
            continue
        best = subset.sort_values('accuracy', ascending=False).iloc[0]
        print(
            f"avg_n≤{limit}: acc={best['accuracy']*100:.2f}%, avg_n={best['avg_n']:.2f}, "
            f"latency={best['avg_latency']:.2f}, shares n1:{best['share_n1']:.2f}, "
            f"n3:{best['share_n3']:.2f}, n5:{best['share_n5']:.2f}, n7:{best['share_n7']:.2f}, "
            f"thresholds=({best['t1']:.2f}, {best['t3']:.2f}, {best['t5']:.2f})"
        )


if __name__ == '__main__':
    main()
