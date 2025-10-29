#!/usr/bin/env python
"""Train lightweight models that predict correctness from prefix entropy trajectories."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_NS = [3, 5, 7]
MAX_TOKENS = 32


@dataclass
class TrainResult:
    target_n: int
    accuracy: float
    auc: float
    scaler_path: Path
    model_path: Path


def parse_entropy_sequences(df: pd.DataFrame, max_tokens: int) -> np.ndarray:
    def to_sequence(raw: str | list) -> List[float]:
        if isinstance(raw, list):
            seq = raw
        else:
            seq = ast.literal_eval(raw) if isinstance(raw, str) and raw else []
        seq = seq[:max_tokens]
        if len(seq) < max_tokens:
            seq = seq + [0.0] * (max_tokens - len(seq))
        return seq

    sequences = df["entropy_tokens"].apply(to_sequence)
    return np.vstack(sequences.to_numpy())


def load_dataset(path: Path, max_tokens: int) -> Tuple[np.ndarray, dict[int, np.ndarray]]:
    df = pd.read_csv(path)
    X = parse_entropy_sequences(df, max_tokens)
    targets = {}
    for n in TARGET_NS:
        targets[n] = df[f"correct_strict_n{n}"].to_numpy()
    return X, targets


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    penalty: str,
    C: float,
    max_iter: int,
    output_dir: Path,
    label: str,
) -> TrainResult:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
    )
    clf.fit(X_train_scaled, y_train)

    probas = clf.predict_proba(X_valid_scaled)[:, 1]
    preds = (probas >= 0.5).astype(int)

    accuracy = accuracy_score(y_valid, preds)
    auc = roc_auc_score(y_valid, probas)

    scaler_path = output_dir / f"entropy_scaler_{label}.json"
    model_path = output_dir / f"entropy_logreg_{label}.npz"

    scaler_info = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with scaler_path.open("w", encoding="utf-8") as fp:
        json.dump(scaler_info, fp)

    np.savez(model_path, coef=clf.coef_, intercept=clf.intercept_)

    return TrainResult(
        target_n=int(label),
        accuracy=float(accuracy),
        auc=float(auc),
        scaler_path=scaler_path,
        model_path=model_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logreg models on entropy trajectories.")
    parser.add_argument(
        "--examples_csv",
        type=Path,
        default=Path("results/suite_20251027_220603/20251029_040055_fixed_examples.csv"),
        help="CSV file with per-example metrics",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/entropy_nn"),
        help="Directory to store models and scalers",
    )
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, targets = load_dataset(args.examples_csv, args.max_tokens)

    report_rows = []
    for n, y in targets.items():
        result = train_model(
            X,
            y,
            seed=args.seed,
            penalty=args.penalty,
            C=args.C,
            max_iter=args.max_iter,
            output_dir=args.output_dir,
            label=str(n),
        )
        report_rows.append(result.__dict__)
        print(
            f"Trained n={n}: accuracy={result.accuracy:.4f}, AUC={result.auc:.4f}, "
            f"model={result.model_path.name}"
        )

    report_df = pd.DataFrame(report_rows)
    report_path = args.output_dir / "training_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"Summary saved to {report_path}")


if __name__ == "__main__":
    main()
