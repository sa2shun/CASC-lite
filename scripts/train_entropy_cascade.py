#!/usr/bin/env python
"""Train stage-wise logistic models for cascade gating based on entropy trajectories."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MAX_TOKENS = 32
N_SPLITS = 5


@dataclass
class StageModel:
    name: str
    accuracy: float
    auc: float
    scaler_path: Path
    model_path: Path


def load_examples(path: Path, max_tokens: int) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)

    def to_sequence(raw: str | List[float]) -> List[float]:
        if isinstance(raw, list):
            seq = raw
        else:
            seq = ast.literal_eval(raw) if isinstance(raw, str) and raw else []
        seq = seq[:max_tokens]
        if len(seq) < max_tokens:
            seq = seq + [0.0] * (max_tokens - len(seq))
        return seq

    entropy_matrix = np.vstack(df["entropy_tokens"].apply(to_sequence).to_numpy())
    return df, entropy_matrix


def train_with_oof(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    penalty: str,
    C: float,
    max_iter: int,
    label: str,
    output_dir: Path,
) -> Tuple[StageModel, np.ndarray]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        clf = LogisticRegression(
            penalty=penalty,
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
        )
        clf.fit(X_train, y[train_idx])
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]

    # train final model on full data
    final_scaler = StandardScaler().fit(X)
    X_scaled = final_scaler.transform(X)
    final_model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
    )
    final_model.fit(X_scaled, y)

    preds = final_model.predict_proba(X_scaled)[:, 1]
    accuracy = accuracy_score(y, (preds >= 0.5).astype(int))
    auc = roc_auc_score(y, preds)

    scaler_path = output_dir / f"cascade_scaler_{label}.json"
    model_path = output_dir / f"cascade_logreg_{label}.npz"

    with scaler_path.open("w", encoding="utf-8") as fp:
        json.dump({"mean": final_scaler.mean_.tolist(), "scale": final_scaler.scale_.tolist()}, fp)

    np.savez(model_path, coef=final_model.coef_, intercept=final_model.intercept_)

    stage_model = StageModel(
        name=label,
        accuracy=float(accuracy),
        auc=float(auc),
        scaler_path=scaler_path,
        model_path=model_path,
    )
    return stage_model, oof_preds


def make_stage2_features(df: pd.DataFrame, entropy_matrix: np.ndarray, oof1: np.ndarray) -> np.ndarray:
    match_13 = (df["canonical_predicted_n1"] == df["canonical_predicted_n3"]).astype(float).to_numpy().reshape(-1, 1)
    return np.hstack([entropy_matrix, oof1.reshape(-1, 1), match_13])


def make_stage3_features(
    df: pd.DataFrame,
    entropy_matrix: np.ndarray,
    oof1: np.ndarray,
    oof2: np.ndarray,
) -> np.ndarray:
    match_13 = (df["canonical_predicted_n1"] == df["canonical_predicted_n3"]).astype(float).to_numpy().reshape(-1, 1)
    match_35 = (df["canonical_predicted_n3"] == df["canonical_predicted_n5"]).astype(float).to_numpy().reshape(-1, 1)
    match_15 = (df["canonical_predicted_n1"] == df["canonical_predicted_n5"]).astype(float).to_numpy().reshape(-1, 1)
    return np.hstack(
        [
            entropy_matrix,
            oof1.reshape(-1, 1),
            oof2.reshape(-1, 1),
            match_13,
            match_35,
            match_15,
        ]
    )


def make_stage4_features(
    df: pd.DataFrame,
    entropy_matrix: np.ndarray,
    oof1: np.ndarray,
    oof2: np.ndarray,
    oof3: np.ndarray,
) -> np.ndarray:
    match_13 = (df["canonical_predicted_n1"] == df["canonical_predicted_n3"]).astype(float).to_numpy().reshape(-1, 1)
    match_35 = (df["canonical_predicted_n3"] == df["canonical_predicted_n5"]).astype(float).to_numpy().reshape(-1, 1)
    match_15 = (df["canonical_predicted_n1"] == df["canonical_predicted_n5"]).astype(float).to_numpy().reshape(-1, 1)
    match_57 = (df["canonical_predicted_n5"] == df["canonical_predicted_n7"]).astype(float).to_numpy().reshape(-1, 1)
    match_37 = (df["canonical_predicted_n3"] == df["canonical_predicted_n7"]).astype(float).to_numpy().reshape(-1, 1)
    match_17 = (df["canonical_predicted_n1"] == df["canonical_predicted_n7"]).astype(float).to_numpy().reshape(-1, 1)
    return np.hstack(
        [
            entropy_matrix,
            oof1.reshape(-1, 1),
            oof2.reshape(-1, 1),
            oof3.reshape(-1, 1),
            match_13,
            match_35,
            match_15,
            match_57,
            match_37,
            match_17,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cascade logistic models using entropy trajectories.")
    parser.add_argument(
        "--examples_csv",
        type=Path,
        default=Path("results/suite_20251027_220603/20251029_040055_fixed_examples.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("models/entropy_cascade"))
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty", type=str, default="l2")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, entropy_matrix = load_examples(args.examples_csv, args.max_tokens)

    targets = {
        "stage1": df["correct_strict_n1"].to_numpy(),
        "stage2": df["correct_strict_n3"].to_numpy(),
        "stage3": df["correct_strict_n5"].to_numpy(),
        "stage4": df["correct_strict_n7"].to_numpy(),
    }

    stage_models: List[StageModel] = []

    # Stage 1
    model1, oof1 = train_with_oof(
        entropy_matrix,
        targets["stage1"],
        seed=args.seed,
        penalty=args.penalty,
        C=args.C,
        max_iter=args.max_iter,
        label="stage1",
        output_dir=args.output_dir,
    )
    stage_models.append(model1)

    # Stage 2
    X_stage2 = make_stage2_features(df, entropy_matrix, oof1)
    model2, oof2 = train_with_oof(
        X_stage2,
        targets["stage2"],
        seed=args.seed,
        penalty=args.penalty,
        C=args.C,
        max_iter=args.max_iter,
        label="stage2",
        output_dir=args.output_dir,
    )
    stage_models.append(model2)

    # Stage 3
    X_stage3 = make_stage3_features(df, entropy_matrix, oof1, oof2)
    model3, oof3 = train_with_oof(
        X_stage3,
        targets["stage3"],
        seed=args.seed,
        penalty=args.penalty,
        C=args.C,
        max_iter=args.max_iter,
        label="stage3",
        output_dir=args.output_dir,
    )
    stage_models.append(model3)

    X_stage4 = make_stage4_features(df, entropy_matrix, oof1, oof2, oof3)

    model4, oof4 = train_with_oof(
        X_stage4,
        targets["stage4"],
        seed=args.seed,
        penalty=args.penalty,
        C=args.C,
        max_iter=args.max_iter,
        label="stage4",
        output_dir=args.output_dir,
    )
    stage_models.append(model4)

    # Save OOF predictions for downstream evaluation
    oof_df = pd.DataFrame(
        {
            "p_stage1": oof1,
            "p_stage2": oof2,
            "p_stage3": oof3,
            "p_stage4": oof4,
        }
    )
    oof_df.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    # Save auxiliary features used in evaluation (match indicators)
    match_df = pd.DataFrame(
        {
            "match_n1_n3": (df["canonical_predicted_n1"] == df["canonical_predicted_n3"]).astype(int),
            "match_n3_n5": (df["canonical_predicted_n3"] == df["canonical_predicted_n5"]).astype(int),
            "match_n5_n7": (df["canonical_predicted_n5"] == df["canonical_predicted_n7"]).astype(int),
            "match_n1_n5": (df["canonical_predicted_n1"] == df["canonical_predicted_n5"]).astype(int),
            "match_n1_n7": (df["canonical_predicted_n1"] == df["canonical_predicted_n7"]).astype(int),
            "match_n3_n7": (df["canonical_predicted_n3"] == df["canonical_predicted_n7"]).astype(int),
        }
    )
    match_df.to_csv(args.output_dir / "aux_features.csv", index=False)

    report = pd.DataFrame(
        [
            {
                "stage": sm.name,
                "accuracy": sm.accuracy,
                "auc": sm.auc,
                "scaler_path": sm.scaler_path.name,
                "model_path": sm.model_path.name,
            }
            for sm in stage_models
        ]
    )
    report.to_csv(args.output_dir / "cascade_training_report.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
