#!/usr/bin/env python3
"""Task-Weighted Ensemble Prediction Merger.

This script combines predictions from multiple models by selecting the best
model's predictions for each task based on historical leaderboard performance.

NOTE: Dec 16, 2025 model weights were lost and cannot be used for ensembling.
      Using next-best available models for tasks where Dec-16 was optimal.

Usage:
    python merge_task_weighted_predictions.py --config task_weighted_ensemble.yaml
    python merge_task_weighted_predictions.py --output submissions/task_weighted_ensemble.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict

import pandas as pd
import yaml  # type: ignore[import-untyped]


class TaskModelMapping(TypedDict):
    """Type definition for task-to-model mapping."""

    submission: str
    model: str
    expected_rank: int


# Task-to-model mapping based on SUBMISSIONS.md analysis
# NOTE: Dec-16 weights lost - using next-best models for KSOL, HLM CLint, MBPB, MGMB
TASK_MODEL_MAPPING: dict[str, TaskModelMapping] = {
    "LogD": {
        "submission": "2026-01-05",
        "model": "chemprop_moe",
        "expected_rank": 15,
    },
    "Log KSOL": {
        "submission": "2026-01-07",  # Dec-16 unavailable, using Jan-07 Large
        "model": "chemprop_large",
        "expected_rank": 15,
    },
    "Log MLM CLint": {
        "submission": "2026-01-05",
        "model": "chemprop_moe",
        "expected_rank": 12,
    },
    "Log HLM CLint": {
        "submission": "2026-01-06",  # Dec-16 unavailable, using Jan-06 Baseline
        "model": "chemprop_baseline",
        "expected_rank": 35,
    },
    "Log Caco-2 Permeability Efflux": {
        "submission": "2026-01-07",
        "model": "chemprop_large",
        "expected_rank": 53,
    },
    "Log Caco-2 Permeability Papp A>B": {
        "submission": "2026-01-07",
        "model": "chemprop_large",
        "expected_rank": 16,
    },
    "Log MPPB": {
        "submission": "2026-01-08",
        "model": "chemeleon_moe",
        "expected_rank": 24,
    },
    "Log MBPB": {
        "submission": "2026-01-05",  # Dec-16 unavailable, using Jan-05 MoE
        "model": "chemprop_moe",
        "expected_rank": 51,
    },
    "Log MGMB": {
        "submission": "2026-01-08",  # Dec-16 unavailable, using Jan-08 Chemeleon
        "model": "chemeleon_moe",
        "expected_rank": 5,
    },
}

# Prediction file paths for each submission (Dec-16 excluded - weights unavailable)
PREDICTION_PATHS = {
    "2026-01-05": (
        "assets/submissions/2026-01-05/mlflow-artifacts/6/"
        "c781fb7efe4a4b70a6fb6263dd3dd8e9/artifacts/predictions/blind_predictions.csv"
    ),
    "2026-01-06": (
        "assets/submissions/2026-01-06/mlflow-artifacts/6/"
        "ca2760b28f5945ee9b387915db9da875/artifacts/predictions/blind_predictions.csv"
    ),
    "2026-01-07": (
        "assets/submissions/2026-01-07/mlflow-artifacts/6/"
        "5ef1d4104f42489184188968ede410d6/artifacts/predictions/blind_predictions.csv"
    ),
    "2026-01-08": (
        "assets/submissions/2026-01-08/mlflow-artifacts/12/"
        "d7d51490fea9458e99e8e6677f425c37/artifacts/predictions/blind_predictions.csv"
    ),
}


def load_predictions(submission_date: str | Path, base_path: Path) -> pd.DataFrame:
    """Load predictions from a submission's cached predictions file.

    Parameters
    ----------
    submission_date : str | Path
        Submission date key or path to predictions file.
    base_path : Path
        Base path for submission directory.

    Returns
    -------
    pd.DataFrame
        Loaded predictions DataFrame.
    """
    submission_date_str = str(submission_date)
    pred_path = base_path / PREDICTION_PATHS[submission_date_str]
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    return pd.read_csv(pred_path)


def merge_task_weighted_predictions(
    base_path: Path,
    output_path: Path,
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """Merge predictions using task-weighted selection strategy.

    For each task, select predictions from the best-performing model.
    """
    # Load all prediction files
    predictions = {}
    submission_dates: set[str] = set(m["submission"] for m in TASK_MODEL_MAPPING.values())
    for date in submission_dates:
        try:
            predictions[date] = load_predictions(date, base_path)
            print(f"✓ Loaded predictions from {date}: {len(predictions[date])} samples")
        except FileNotFoundError as e:
            print(f"✗ {e}")
            return None

    # Use the first prediction file as the base (for SMILES column)
    base_date = list(predictions.keys())[0]
    merged = predictions[base_date][[smiles_col]].copy()

    # Select best model's predictions for each task
    print("\nTask-Model Assignment:")
    print("-" * 60)
    for task, mapping in TASK_MODEL_MAPPING.items():
        submission = mapping["submission"]
        model = mapping["model"]
        rank = mapping["expected_rank"]

        if task in predictions[submission].columns:
            merged[task] = predictions[submission][task]
            print(f"  {task:35s} ← {submission} ({model}, rank #{rank})")
        else:
            print(f"  {task:35s} ✗ Column not found in {submission}")

    # Save merged predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n✓ Saved merged predictions to: {output_path}")
    print(f"  Shape: {merged.shape}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge predictions using task-weighted selection strategy")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("."),
        help="Base path to the project root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/submissions/task_weighted_ensemble/blind_predictions.csv"),
        help="Output path for merged predictions",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to task_weighted_ensemble.yaml config (optional)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Task-Weighted Ensemble Prediction Merger")
    print("=" * 60)
    print()

    # Load config if provided
    if args.config and args.config.exists():
        with open(args.config) as f:
            yaml.safe_load(f)  # Config not currently used, but available for future
        print(f"Loaded config from: {args.config}")

    # Merge predictions
    merged = merge_task_weighted_predictions(
        base_path=args.base_path,
        output_path=args.output,
    )

    if merged is not None:
        print("\n" + "=" * 60)
        print("Summary Statistics:")
        print("=" * 60)
        for col in merged.columns:
            if col != "SMILES":
                print(f"  {col:35s}: mean={merged[col].mean():.4f}, std={merged[col].std():.4f}")


if __name__ == "__main__":
    main()
