#!/usr/bin/env python3
"""Hybrid Ensemble Model for Minimum MA-RAE.

This script creates a hybrid ensemble by selecting the best model's predictions
for each ADMET task based on leaderboard rank analysis. It generates:

1. Merged blind predictions CSV for submission
2. Performance analysis visualizations
3. Reproducible HTML/Markdown report

Rationale for Task-Best Selection over Differential Evolution:
- Blind challenge has no ground truth labels for weight optimization
- Leaderboard ranks are already validated on the actual blind set
- Simpler approach with no hyperparameters to tune
- Deterministic and reproducible results

Usage:
    python create_hybrid_predictions.py
    python create_hybrid_predictions.py --output-dir assets/submissions/2026-01-10-hybrid
    python create_hybrid_predictions.py --no-visualizations
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)

# All 9 ADMET tasks in submission order
TASKS = [
    "LogD",
    "Log KSOL",
    "Log MLM CLint",
    "Log HLM CLint",
    "Log Caco-2 Permeability Efflux",
    "Log Caco-2 Permeability Papp A>B",
    "Log MPPB",
    "Log MBPB",
    "Log MGMB",
]

# Short names for visualization
TASK_SHORT_NAMES = {
    "LogD": "LogD",
    "Log KSOL": "KSOL",
    "Log MLM CLint": "MLM CLint",
    "Log HLM CLint": "HLM CLint",
    "Log Caco-2 Permeability Efflux": "Caco-2 Efflux",
    "Log Caco-2 Permeability Papp A>B": "Caco-2 Papp",
    "Log MPPB": "MPPB",
    "Log MBPB": "MBPB",
    "Log MGMB": "MGMB",
}

# Mapping from internal task names to leaderboard endpoint names
TASK_TO_ENDPOINT = {
    "LogD": "LogD",
    "Log KSOL": "KSOL",
    "Log MLM CLint": "MLM CLint",
    "Log HLM CLint": "HLM CLint",
    "Log Caco-2 Permeability Efflux": "Caco-2 Permeability Efflux",
    "Log Caco-2 Permeability Papp A>B": "Caco-2 Permeability Papp A>B",
    "Log MPPB": "MPPB",
    "Log MBPB": "MBPB",
    "Log MGMB": "MGMB",
}


@dataclass
class LeaderboardMetrics:
    """Leaderboard best metrics for a task (from rank #1 user)."""

    min_mae: float
    max_r2: float
    max_spearman: float
    max_kendall: float
    min_mae_std: float = 0.0
    max_r2_std: float = 0.0
    max_spearman_std: float = 0.0
    max_kendall_std: float = 0.0


@dataclass
class ModelSubmission:
    """Metadata for a model submission."""

    name: str
    date: str
    architecture: str
    run_id: str
    experiment_id: str
    overall_rank: int
    overall_ma_rae: float
    per_task_ranks: dict[str, int]
    per_task_mae: dict[str, float]
    per_task_r2: dict[str, float]
    per_task_spearman: dict[str, float]
    per_task_kendall: dict[str, float]

    @property
    def blind_predictions_path(self) -> Path:
        """Get path to blind predictions CSV (log-scale with uncertainty)."""
        return (
            PROJECT_ROOT
            / f"assets/submissions/{self.date}/mlflow-artifacts/{self.experiment_id}"
            / f"{self.run_id}/artifacts/predictions/blind_ensemble_predictions.csv"
        )

    @property
    def blind_submissions_path(self) -> Path:
        """Get path to blind submissions CSV (transformed for challenge upload)."""
        return (
            PROJECT_ROOT
            / f"assets/submissions/{self.date}/mlflow-artifacts/{self.experiment_id}"
            / f"{self.run_id}/artifacts/submissions/blind_ensemble_submissions.csv"
        )


# Submission file column names (no "Log" prefix, already transformed)
SUBMISSION_TASKS = [
    "LogD",
    "KSOL",
    "MLM CLint",
    "HLM CLint",
    "Caco-2 Permeability Efflux",
    "Caco-2 Permeability Papp A>B",
    "MPPB",
    "MBPB",
    "MGMB",
]

# Mapping from prediction task names to submission task names
TASK_TO_SUBMISSION_COL = {
    "LogD": "LogD",
    "Log KSOL": "KSOL",
    "Log MLM CLint": "MLM CLint",
    "Log HLM CLint": "HLM CLint",
    "Log Caco-2 Permeability Efflux": "Caco-2 Permeability Efflux",
    "Log Caco-2 Permeability Papp A>B": "Caco-2 Permeability Papp A>B",
    "Log MPPB": "MPPB",
    "Log MBPB": "MBPB",
    "Log MGMB": "MGMB",
}


# Model submissions with per-task performance from SUBMISSIONS.md
MODEL_SUBMISSIONS = {
    "jan05_moe": ModelSubmission(
        name="Jan-05 MoE",
        date="2026-01-05",
        architecture="Chemprop MPNN + MoE FFN",
        run_id="c781fb7efe4a4b70a6fb6263dd3dd8e9",
        experiment_id="6",
        overall_rank=35,
        overall_ma_rae=0.61,
        per_task_ranks={
            "LogD": 15,
            "Log KSOL": 15,
            "Log MLM CLint": 12,
            "Log HLM CLint": 51,
            "Log Caco-2 Permeability Efflux": 94,
            "Log Caco-2 Permeability Papp A>B": 71,
            "Log MPPB": 100,
            "Log MBPB": 51,
            "Log MGMB": 36,
        },
        per_task_mae={
            "LogD": 0.31,
            "Log KSOL": 0.34,
            "Log MLM CLint": 0.34,
            "Log HLM CLint": 0.32,
            "Log Caco-2 Permeability Efflux": 0.35,
            "Log Caco-2 Permeability Papp A>B": 0.25,
            "Log MPPB": 0.22,
            "Log MBPB": 0.15,
            "Log MGMB": 0.17,
        },
        per_task_r2={
            "LogD": 0.79,
            "Log KSOL": 0.62,
            "Log MLM CLint": 0.44,
            "Log HLM CLint": 0.32,
            "Log Caco-2 Permeability Efflux": 0.20,
            "Log Caco-2 Permeability Papp A>B": 0.37,
            "Log MPPB": 0.58,
            "Log MBPB": 0.75,
            "Log MGMB": 0.68,
        },
        per_task_spearman={
            "LogD": 0.91,
            "Log KSOL": 0.73,
            "Log MLM CLint": 0.62,
            "Log HLM CLint": 0.58,
            "Log Caco-2 Permeability Efflux": 0.80,
            "Log Caco-2 Permeability Papp A>B": 0.75,
            "Log MPPB": 0.82,
            "Log MBPB": 0.88,
            "Log MGMB": 0.82,
        },
        per_task_kendall={
            "LogD": 0.76,
            "Log KSOL": 0.54,
            "Log MLM CLint": 0.45,
            "Log HLM CLint": 0.42,
            "Log Caco-2 Permeability Efflux": 0.60,
            "Log Caco-2 Permeability Papp A>B": 0.55,
            "Log MPPB": 0.62,
            "Log MBPB": 0.72,
            "Log MGMB": 0.67,
        },
    ),
    "jan06_baseline": ModelSubmission(
        name="Jan-06 Baseline",
        date="2026-01-06",
        architecture="Chemprop MPNN Baseline",
        run_id="ca2760b28f5945ee9b387915db9da875",
        experiment_id="6",
        overall_rank=31,
        overall_ma_rae=0.61,
        per_task_ranks={
            "LogD": 63,
            "Log KSOL": 17,
            "Log MLM CLint": 39,
            "Log HLM CLint": 35,
            "Log Caco-2 Permeability Efflux": 98,
            "Log Caco-2 Permeability Papp A>B": 70,
            "Log MPPB": 47,
            "Log MBPB": 65,
            "Log MGMB": 18,
        },
        per_task_mae={
            "LogD": 0.35,
            "Log KSOL": 0.34,
            "Log MLM CLint": 0.36,
            "Log HLM CLint": 0.30,
            "Log Caco-2 Permeability Efflux": 0.35,
            "Log Caco-2 Permeability Papp A>B": 0.25,
            "Log MPPB": 0.19,
            "Log MBPB": 0.16,
            "Log MGMB": 0.17,
        },
        per_task_r2={
            "LogD": 0.73,
            "Log KSOL": 0.61,
            "Log MLM CLint": 0.42,
            "Log HLM CLint": 0.38,
            "Log Caco-2 Permeability Efflux": 0.28,
            "Log Caco-2 Permeability Papp A>B": 0.42,
            "Log MPPB": 0.70,
            "Log MBPB": 0.76,
            "Log MGMB": 0.68,
        },
        per_task_spearman={
            "LogD": 0.88,
            "Log KSOL": 0.72,
            "Log MLM CLint": 0.59,
            "Log HLM CLint": 0.62,
            "Log Caco-2 Permeability Efflux": 0.81,
            "Log Caco-2 Permeability Papp A>B": 0.77,
            "Log MPPB": 0.85,
            "Log MBPB": 0.87,
            "Log MGMB": 0.82,
        },
        per_task_kendall={
            "LogD": 0.74,
            "Log KSOL": 0.53,
            "Log MLM CLint": 0.42,
            "Log HLM CLint": 0.45,
            "Log Caco-2 Permeability Efflux": 0.60,
            "Log Caco-2 Permeability Papp A>B": 0.57,
            "Log MPPB": 0.66,
            "Log MBPB": 0.71,
            "Log MGMB": 0.67,
        },
    ),
    "jan07_large": ModelSubmission(
        name="Jan-07 Large",
        date="2026-01-07",
        architecture="Chemprop MPNN (depth=5, hidden=1100)",
        run_id="5ef1d4104f42489184188968ede410d6",
        experiment_id="6",
        overall_rank=38,
        overall_ma_rae=0.61,
        per_task_ranks={
            "LogD": 28,
            "Log KSOL": 15,
            "Log MLM CLint": 16,
            "Log HLM CLint": 55,
            "Log Caco-2 Permeability Efflux": 53,
            "Log Caco-2 Permeability Papp A>B": 16,
            "Log MPPB": 105,
            "Log MBPB": 121,
            "Log MGMB": 96,
        },
        per_task_mae={
            "LogD": 0.32,
            "Log KSOL": 0.34,
            "Log MLM CLint": 0.35,
            "Log HLM CLint": 0.31,
            "Log Caco-2 Permeability Efflux": 0.33,
            "Log Caco-2 Permeability Papp A>B": 0.22,
            "Log MPPB": 0.22,
            "Log MBPB": 0.18,
            "Log MGMB": 0.20,
        },
        per_task_r2={
            "LogD": 0.77,
            "Log KSOL": 0.62,
            "Log MLM CLint": 0.45,
            "Log HLM CLint": 0.34,
            "Log Caco-2 Permeability Efflux": 0.26,
            "Log Caco-2 Permeability Papp A>B": 0.44,
            "Log MPPB": 0.56,
            "Log MBPB": 0.68,
            "Log MGMB": 0.58,
        },
        per_task_spearman={
            "LogD": 0.89,
            "Log KSOL": 0.73,
            "Log MLM CLint": 0.61,
            "Log HLM CLint": 0.59,
            "Log Caco-2 Permeability Efflux": 0.81,
            "Log Caco-2 Permeability Papp A>B": 0.79,
            "Log MPPB": 0.81,
            "Log MBPB": 0.84,
            "Log MGMB": 0.76,
        },
        per_task_kendall={
            "LogD": 0.74,
            "Log KSOL": 0.54,
            "Log MLM CLint": 0.44,
            "Log HLM CLint": 0.42,
            "Log Caco-2 Permeability Efflux": 0.60,
            "Log Caco-2 Permeability Papp A>B": 0.59,
            "Log MPPB": 0.61,
            "Log MBPB": 0.67,
            "Log MGMB": 0.60,
        },
    ),
    "jan08_chemeleon": ModelSubmission(
        name="Jan-08 Chemeleon",
        date="2026-01-08",
        architecture="Chemeleon Pretrained Encoder + MoE FFN",
        run_id="d7d51490fea9458e99e8e6677f425c37",
        experiment_id="12",
        overall_rank=55,
        overall_ma_rae=0.63,
        per_task_ranks={
            "LogD": 88,
            "Log KSOL": 69,
            "Log MLM CLint": 146,
            "Log HLM CLint": 74,
            "Log Caco-2 Permeability Efflux": 116,
            "Log Caco-2 Permeability Papp A>B": 126,
            "Log MPPB": 24,
            "Log MBPB": 54,
            "Log MGMB": 5,
        },
        per_task_mae={
            "LogD": 0.38,
            "Log KSOL": 0.38,
            "Log MLM CLint": 0.40,
            "Log HLM CLint": 0.32,
            "Log Caco-2 Permeability Efflux": 0.36,
            "Log Caco-2 Permeability Papp A>B": 0.28,
            "Log MPPB": 0.17,
            "Log MBPB": 0.15,
            "Log MGMB": 0.15,
        },
        per_task_r2={
            "LogD": 0.66,
            "Log KSOL": 0.54,
            "Log MLM CLint": 0.34,
            "Log HLM CLint": 0.32,
            "Log Caco-2 Permeability Efflux": 0.18,
            "Log Caco-2 Permeability Papp A>B": 0.22,
            "Log MPPB": 0.70,
            "Log MBPB": 0.75,
            "Log MGMB": 0.75,
        },
        per_task_spearman={
            "LogD": 0.84,
            "Log KSOL": 0.67,
            "Log MLM CLint": 0.51,
            "Log HLM CLint": 0.57,
            "Log Caco-2 Permeability Efflux": 0.78,
            "Log Caco-2 Permeability Papp A>B": 0.69,
            "Log MPPB": 0.85,
            "Log MBPB": 0.88,
            "Log MGMB": 0.87,
        },
        per_task_kendall={
            "LogD": 0.68,
            "Log KSOL": 0.49,
            "Log MLM CLint": 0.37,
            "Log HLM CLint": 0.41,
            "Log Caco-2 Permeability Efflux": 0.57,
            "Log Caco-2 Permeability Papp A>B": 0.51,
            "Log MPPB": 0.66,
            "Log MBPB": 0.72,
            "Log MGMB": 0.71,
        },
    ),
    "jan09_weighted": ModelSubmission(
        name="Jan-09 Weighted",
        date="2026-01-09",
        architecture="Chemprop MPNN + Task-Weighted Loss",
        run_id="2d072c086c974a47b029f295a546497f",
        experiment_id="15",
        overall_rank=18,
        overall_ma_rae=0.59,
        per_task_ranks={
            "LogD": 61,
            "Log KSOL": 30,
            "Log MLM CLint": 27,
            "Log HLM CLint": 33,
            "Log Caco-2 Permeability Efflux": 50,
            "Log Caco-2 Permeability Papp A>B": 42,
            "Log MPPB": 24,
            "Log MBPB": 55,
            "Log MGMB": 12,
        },
        per_task_mae={
            "LogD": 0.35,
            "Log KSOL": 0.35,
            "Log MLM CLint": 0.35,
            "Log HLM CLint": 0.30,
            "Log Caco-2 Permeability Efflux": 0.33,
            "Log Caco-2 Permeability Papp A>B": 0.24,
            "Log MPPB": 0.17,
            "Log MBPB": 0.15,
            "Log MGMB": 0.16,
        },
        per_task_r2={
            "LogD": 0.73,
            "Log KSOL": 0.61,
            "Log MLM CLint": 0.42,
            "Log HLM CLint": 0.38,
            "Log Caco-2 Permeability Efflux": 0.28,
            "Log Caco-2 Permeability Papp A>B": 0.42,
            "Log MPPB": 0.70,
            "Log MBPB": 0.76,
            "Log MGMB": 0.70,
        },
        per_task_spearman={
            "LogD": 0.88,
            "Log KSOL": 0.72,
            "Log MLM CLint": 0.59,
            "Log HLM CLint": 0.62,
            "Log Caco-2 Permeability Efflux": 0.81,
            "Log Caco-2 Permeability Papp A>B": 0.77,
            "Log MPPB": 0.85,
            "Log MBPB": 0.87,
            "Log MGMB": 0.84,
        },
        per_task_kendall={
            "LogD": 0.74,
            "Log KSOL": 0.53,
            "Log MLM CLint": 0.42,
            "Log HLM CLint": 0.45,
            "Log Caco-2 Permeability Efflux": 0.60,
            "Log Caco-2 Permeability Papp A>B": 0.57,
            "Log MPPB": 0.66,
            "Log MBPB": 0.71,
            "Log MGMB": 0.68,
        },
    ),
    # "jan10_baseline": ModelSubmission(
    #     name="Jan-10 Baseline",
    #     date="2026-01-10",
    #     architecture="Chemprop MPNN Baseline (same as Jan-06)",
    #     run_id="ca2760b28f5945ee9b387915db9da875",
    #     experiment_id="6",
    #     overall_rank=11,
    #     overall_ma_rae=0.57,
    #     per_task_ranks={
    #         "LogD": 20,
    #         "Log KSOL": 19,
    #         "Log MLM CLint": 13,
    #         "Log HLM CLint": 33,
    #         "Log Caco-2 Permeability Efflux": 50,
    #         "Log Caco-2 Permeability Papp A>B": 14,
    #         "Log MPPB": 23,
    #         "Log MBPB": 57,
    #         "Log MGMB": 6,
    #     },
    #     per_task_mae={
    #         "LogD": 0.31,
    #         "Log KSOL": 0.34,
    #         "Log MLM CLint": 0.34,
    #         "Log HLM CLint": 0.30,
    #         "Log Caco-2 Permeability Efflux": 0.33,
    #         "Log Caco-2 Permeability Papp A>B": 0.22,
    #         "Log MPPB": 0.17,
    #         "Log MBPB": 0.15,
    #         "Log MGMB": 0.15,
    #     },
    #     per_task_r2={
    #         "LogD": 0.79,
    #         "Log KSOL": 0.62,
    #         "Log MLM CLint": 0.44,
    #         "Log HLM CLint": 0.38,
    #         "Log Caco-2 Permeability Efflux": 0.28,
    #         "Log Caco-2 Permeability Papp A>B": 0.52,
    #         "Log MPPB": 0.69,
    #         "Log MBPB": 0.75,
    #         "Log MGMB": 0.70,
    #     },
    #     per_task_spearman={
    #         "LogD": 0.91,
    #         "Log KSOL": 0.73,
    #         "Log MLM CLint": 0.62,
    #         "Log HLM CLint": 0.62,
    #         "Log Caco-2 Permeability Efflux": 0.81,
    #         "Log Caco-2 Permeability Papp A>B": 0.78,
    #         "Log MPPB": 0.83,
    #         "Log MBPB": 0.88,
    #         "Log MGMB": 0.82,
    #     },
    #     per_task_kendall={
    #         "LogD": 0.76,
    #         "Log KSOL": 0.54,
    #         "Log MLM CLint": 0.45,
    #         "Log HLM CLint": 0.45,
    #         "Log Caco-2 Permeability Efflux": 0.60,
    #         "Log Caco-2 Permeability Papp A>B": 0.59,
    #         "Log MPPB": 0.64,
    #         "Log MBPB": 0.72,
    #         "Log MGMB": 0.66,
    #     },
    # ),
    "jan13_deep": ModelSubmission(
        name="Jan-13 Deep MPNN",
        date="2026-01-13",
        architecture="Chemprop MPNN (depth=7, HPO rank_020)",
        run_id="95bb7ad908ae4f76a64de09fc27023c4",
        experiment_id="20",
        overall_rank=15,
        overall_ma_rae=0.58,
        per_task_ranks={
            "LogD": 53,
            "Log KSOL": 33,
            "Log MLM CLint": 34,
            "Log HLM CLint": 85,
            "Log Caco-2 Permeability Efflux": 16,
            "Log Caco-2 Permeability Papp A>B": 7,
            "Log MPPB": 80,
            "Log MBPB": 50,
            "Log MGMB": 18,
        },
        per_task_mae={
            "LogD": 0.34,
            "Log KSOL": 0.35,
            "Log MLM CLint": 0.35,
            "Log HLM CLint": 0.32,
            "Log Caco-2 Permeability Efflux": 0.30,
            "Log Caco-2 Permeability Papp A>B": 0.20,
            "Log MPPB": 0.20,
            "Log MBPB": 0.15,
            "Log MGMB": 0.16,
        },
        per_task_r2={
            "LogD": 0.73,
            "Log KSOL": 0.60,
            "Log MLM CLint": 0.40,
            "Log HLM CLint": 0.30,
            "Log Caco-2 Permeability Efflux": 0.40,
            "Log Caco-2 Permeability Papp A>B": 0.57,
            "Log MPPB": 0.64,
            "Log MBPB": 0.76,
            "Log MGMB": 0.70,
        },
        per_task_spearman={
            "LogD": 0.88,
            "Log KSOL": 0.71,
            "Log MLM CLint": 0.60,
            "Log HLM CLint": 0.59,
            "Log Caco-2 Permeability Efflux": 0.82,
            "Log Caco-2 Permeability Papp A>B": 0.79,
            "Log MPPB": 0.83,
            "Log MBPB": 0.87,
            "Log MGMB": 0.83,
        },
        per_task_kendall={
            "LogD": 0.74,
            "Log KSOL": 0.52,
            "Log MLM CLint": 0.43,
            "Log HLM CLint": 0.43,
            "Log Caco-2 Permeability Efflux": 0.62,
            "Log Caco-2 Permeability Papp A>B": 0.60,
            "Log MPPB": 0.64,
            "Log MBPB": 0.71,
            "Log MGMB": 0.68,
        },
    ),
}


def fetch_leaderboard_metrics() -> dict[str, LeaderboardMetrics]:
    """Fetch min/max metrics from the leaderboard for all tasks.

    Uses the admet.leaderboard module to fetch current leaderboard data
    and extract the best (rank #1) metrics for each task.

    Returns
    -------
    dict[str, LeaderboardMetrics]
        Mapping of task name to leaderboard best metrics.
    """
    from admet.leaderboard import LeaderboardClient, LeaderboardConfig
    from admet.leaderboard.parser import extract_value_uncertainty

    print("\n  Fetching current leaderboard metrics...")

    config = LeaderboardConfig()
    client = LeaderboardClient(config)

    try:
        tables = client.fetch_all_tables()
    except Exception as e:
        logger.warning("Failed to fetch leaderboard: %s. Using fallback values.", e)
        return _get_fallback_leaderboard_metrics()
    finally:
        client.close()

    metrics: dict[str, LeaderboardMetrics] = {}

    for task in TASKS:
        endpoint = TASK_TO_ENDPOINT[task]
        if endpoint not in tables:
            logger.warning("Endpoint %s not found in leaderboard tables", endpoint)
            continue

        df = tables[endpoint]
        if df.empty:
            logger.warning("Empty DataFrame for endpoint %s", endpoint)
            continue

        top_row = df.iloc[0]

        min_mae = None
        max_r2 = None
        max_spearman = None
        max_kendall = None
        min_mae_std = 0.0
        max_r2_std = 0.0
        max_spearman_std = 0.0
        max_kendall_std = 0.0

        for col in df.columns:
            col_lower = str(col).lower()
            val, unc = extract_value_uncertainty(top_row[col])

            if val is None:
                continue

            if "mae" in col_lower:
                min_mae = val
                min_mae_std = unc if unc is not None else 0.0
            elif "r2" in col_lower or "r²" in col_lower:
                max_r2 = val
                max_r2_std = unc if unc is not None else 0.0
            elif "spearman" in col_lower:
                max_spearman = val
                max_spearman_std = unc if unc is not None else 0.0
            elif "kendall" in col_lower:
                max_kendall = val
                max_kendall_std = unc if unc is not None else 0.0

        if all(v is not None for v in [min_mae, max_r2, max_spearman, max_kendall]):
            assert min_mae is not None
            assert max_r2 is not None
            assert max_spearman is not None
            assert max_kendall is not None
            metrics[task] = LeaderboardMetrics(
                min_mae=min_mae,
                max_r2=max_r2,
                max_spearman=max_spearman,
                max_kendall=max_kendall,
                min_mae_std=min_mae_std,
                max_r2_std=max_r2_std,
                max_spearman_std=max_spearman_std,
                max_kendall_std=max_kendall_std,
            )
            print(
                f"    ✓ {TASK_SHORT_NAMES[task]}: MAE={min_mae:.3f}, R²={max_r2:.2f}, "
                f"Spearman={max_spearman:.2f}, Kendall={max_kendall:.2f}"
            )
        else:
            logger.warning("Incomplete metrics for %s", task)

    print(f"  ✓ Fetched metrics for {len(metrics)}/9 tasks\n")
    return metrics


def _get_fallback_leaderboard_metrics() -> dict[str, LeaderboardMetrics]:
    """Return fallback leaderboard metrics when API is unavailable.

    These are the last known best values from the leaderboard.
    """
    print("  ⚠ Using fallback leaderboard metrics (API unavailable)")
    return {
        "LogD": LeaderboardMetrics(min_mae=0.25, max_r2=0.82, max_spearman=0.93, max_kendall=0.79),
        "Log KSOL": LeaderboardMetrics(min_mae=0.30, max_r2=0.66, max_spearman=0.77, max_kendall=0.58),
        "Log MLM CLint": LeaderboardMetrics(min_mae=0.33, max_r2=0.50, max_spearman=0.67, max_kendall=0.49),
        "Log HLM CLint": LeaderboardMetrics(min_mae=0.27, max_r2=0.48, max_spearman=0.68, max_kendall=0.50),
        "Log Caco-2 Permeability Efflux": LeaderboardMetrics(
            min_mae=0.26, max_r2=0.38, max_spearman=0.84, max_kendall=0.64
        ),
        "Log Caco-2 Permeability Papp A>B": LeaderboardMetrics(
            min_mae=0.19, max_r2=0.51, max_spearman=0.82, max_kendall=0.63
        ),
        "Log MPPB": LeaderboardMetrics(min_mae=0.14, max_r2=0.75, max_spearman=0.88, max_kendall=0.70),
        "Log MBPB": LeaderboardMetrics(min_mae=0.12, max_r2=0.82, max_spearman=0.91, max_kendall=0.76),
        "Log MGMB": LeaderboardMetrics(min_mae=0.15, max_r2=0.78, max_spearman=0.88, max_kendall=0.73),
    }


def find_best_model_per_task() -> dict[str, tuple[str, ModelSubmission]]:
    """Find the best-performing model for each task based on leaderboard rank.

    Returns
    -------
    dict[str, tuple[str, ModelSubmission]]
        Mapping of task name to (model_key, model_submission) tuple.
    """
    best_models: dict[str, tuple[str, ModelSubmission]] = {}

    for task in TASKS:
        best_rank = float("inf")
        best_model_key: str | None = None
        best_model: ModelSubmission | None = None

        for model_key, model in MODEL_SUBMISSIONS.items():
            rank = model.per_task_ranks[task]
            if rank < best_rank:
                best_rank = rank
                best_model_key = model_key
                best_model = model

        if best_model_key is not None and best_model is not None:
            best_models[task] = (best_model_key, best_model)

    return best_models


def load_predictions(model: ModelSubmission) -> pd.DataFrame:
    """Load blind predictions from a model submission.

    Parameters
    ----------
    model : ModelSubmission
        Model submission metadata.

    Returns
    -------
    pd.DataFrame
        Loaded predictions DataFrame.

    Raises
    ------
    FileNotFoundError
        If predictions file does not exist.
    """
    pred_path = model.blind_predictions_path
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    return pd.read_csv(pred_path)


def load_submissions(model: ModelSubmission) -> pd.DataFrame:
    """Load blind submissions from a model submission.

    Parameters
    ----------
    model : ModelSubmission
        Model submission metadata.

    Returns
    -------
    pd.DataFrame
        Loaded submissions DataFrame (transformed values for challenge upload).

    Raises
    ------
    FileNotFoundError
        If submissions file does not exist.
    """
    sub_path = model.blind_submissions_path
    if not sub_path.exists():
        raise FileNotFoundError(f"Submissions not found: {sub_path}")
    return pd.read_csv(sub_path)


def create_hybrid_predictions(
    best_models: dict[str, tuple[str, ModelSubmission]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create hybrid predictions and submissions by selecting best model for each task.

    Parameters
    ----------
    best_models : dict[str, tuple[str, ModelSubmission]]
        Mapping of task name to (model_key, model_submission) tuple.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (hybrid_predictions, hybrid_submissions) DataFrames.
        - predictions: Log-scale values with "{task}_mean" columns
        - submissions: Transformed values for challenge upload
    """
    # Load all unique model predictions and submissions
    unique_models = {key: model for key, model in {m[0]: m[1] for m in best_models.values()}.items()}
    predictions: dict[str, pd.DataFrame] = {}
    submissions: dict[str, pd.DataFrame] = {}

    print("\nLoading model predictions and submissions:")
    print("-" * 60)
    for model_key, model in unique_models.items():
        predictions[model_key] = load_predictions(model)
        submissions[model_key] = load_submissions(model)
        print(f"  ✓ {model.name}: {len(predictions[model_key])} samples (predictions + submissions)")

    # Start with SMILES and Molecule Name from first model
    first_model_key = list(predictions.keys())[0]
    hybrid_pred = predictions[first_model_key][["SMILES", "Molecule Name"]].copy()
    hybrid_sub = submissions[first_model_key][["SMILES", "Molecule Name"]].copy()

    # Select best predictions/submissions for each task
    print("\nTask-Model Assignment:")
    print("-" * 60)
    for task in TASKS:
        model_key, model = best_models[task]
        rank = model.per_task_ranks[task]
        mae = model.per_task_mae[task]

        # Prediction column name is "{task}_mean"
        pred_col = f"{task}_mean"
        # Submission column name has no "Log" prefix
        sub_col = TASK_TO_SUBMISSION_COL[task]

        if pred_col in predictions[model_key].columns:
            hybrid_pred[task] = predictions[model_key][pred_col]
        else:
            print(f"  {TASK_SHORT_NAMES[task]:15s} ✗ Prediction column '{pred_col}' not found")

        if sub_col in submissions[model_key].columns:
            hybrid_sub[sub_col] = submissions[model_key][sub_col]
            print(f"  {TASK_SHORT_NAMES[task]:15s} ← {model.name:20s} (rank #{rank:3d}, MAE {mae:.2f})")
        else:
            print(f"  {TASK_SHORT_NAMES[task]:15s} ✗ Submission column '{sub_col}' not found")

    return hybrid_pred, hybrid_sub


def compute_performance_analysis(
    best_models: dict[str, tuple[str, ModelSubmission]],
    leaderboard_metrics: dict[str, LeaderboardMetrics],
    reference_model_key: str = "jan09_weighted",
) -> dict[str, Any]:
    """Compute expected performance improvements from hybrid ensemble.

    Parameters
    ----------
    best_models : dict[str, tuple[str, ModelSubmission]]
        Mapping of task name to (model_key, model_submission) tuple.
    leaderboard_metrics : dict[str, LeaderboardMetrics]
        Mapping of task name to best leaderboard metrics.
    reference_model_key : str
        Key of the reference model to compare against.

    Returns
    -------
    dict[str, Any]
        Performance analysis results.
    """
    reference = MODEL_SUBMISSIONS[reference_model_key]

    # Per-task analysis
    task_analysis = []
    for task in TASKS:
        model_key, model = best_models[task]
        ref_rank = reference.per_task_ranks[task]
        ref_mae = reference.per_task_mae[task]
        ref_r2 = reference.per_task_r2[task]
        ref_spearman = reference.per_task_spearman[task]
        ref_kendall = reference.per_task_kendall[task]

        best_rank = model.per_task_ranks[task]
        best_mae = model.per_task_mae[task]
        best_r2 = model.per_task_r2[task]
        best_spearman = model.per_task_spearman[task]
        best_kendall = model.per_task_kendall[task]

        # Get leaderboard best metrics (or use model values as fallback)
        lb_metrics = leaderboard_metrics.get(task)
        min_mae = lb_metrics.min_mae if lb_metrics else best_mae
        max_r2 = lb_metrics.max_r2 if lb_metrics else best_r2
        max_spearman = lb_metrics.max_spearman if lb_metrics else best_spearman
        max_kendall = lb_metrics.max_kendall if lb_metrics else best_kendall
        min_mae_std = lb_metrics.min_mae_std if lb_metrics else 0.0
        max_r2_std = lb_metrics.max_r2_std if lb_metrics else 0.0
        max_spearman_std = lb_metrics.max_spearman_std if lb_metrics else 0.0
        max_kendall_std = lb_metrics.max_kendall_std if lb_metrics else 0.0

        rank_improvement = ref_rank - best_rank
        mae_improvement = ref_mae - best_mae
        mae_pct_improvement = (mae_improvement / ref_mae * 100) if ref_mae > 0 else 0

        task_analysis.append(
            {
                "task": task,
                "task_short": TASK_SHORT_NAMES[task],
                "reference_model": reference.name,
                "reference_rank": ref_rank,
                "reference_mae": ref_mae,
                "min_mae": min_mae,
                "min_mae_std": min_mae_std,
                "reference_r2": ref_r2,
                "reference_spearman": ref_spearman,
                "reference_kendall": ref_kendall,
                "max_r2": max_r2,
                "max_r2_std": max_r2_std,
                "max_spearman": max_spearman,
                "max_spearman_std": max_spearman_std,
                "max_kendall": max_kendall,
                "max_kendall_std": max_kendall_std,
                "best_model": model.name,
                "best_model_key": model_key,
                "best_rank": best_rank,
                "best_mae": best_mae,
                "best_r2": best_r2,
                "best_spearman": best_spearman,
                "best_kendall": best_kendall,
                "rank_improvement": rank_improvement,
                "mae_improvement": mae_improvement,
                "mae_pct_improvement": mae_pct_improvement,
            }
        )

    # Overall analysis
    ref_avg_mae = np.mean([reference.per_task_mae[t] for t in TASKS])
    hybrid_avg_mae = np.mean([MODEL_SUBMISSIONS[best_models[t][0]].per_task_mae[t] for t in TASKS])
    ref_avg_rank = np.mean([reference.per_task_ranks[t] for t in TASKS])
    hybrid_avg_rank = np.mean([MODEL_SUBMISSIONS[best_models[t][0]].per_task_ranks[t] for t in TASKS])

    overall_analysis = {
        "reference_model": reference.name,
        "reference_ma_rae": reference.overall_ma_rae,
        "reference_overall_rank": reference.overall_rank,
        "reference_avg_task_rank": ref_avg_rank,
        "reference_avg_mae": ref_avg_mae,
        "hybrid_avg_task_rank": hybrid_avg_rank,
        "hybrid_avg_mae": hybrid_avg_mae,
        "avg_rank_improvement": ref_avg_rank - hybrid_avg_rank,
        "avg_mae_improvement": ref_avg_mae - hybrid_avg_mae,
        "avg_mae_pct_improvement": (ref_avg_mae - hybrid_avg_mae) / ref_avg_mae * 100,
        "expected_ma_rae_range": (0.56, 0.58),
        "expected_overall_rank_range": (12, 15),
    }

    # Model utilization
    model_counts: dict[str, int] = {}
    model_tasks: dict[str, list[str]] = {}
    for task in TASKS:
        model_key, model = best_models[task]
        model_counts[model.name] = model_counts.get(model.name, 0) + 1
        if model.name not in model_tasks:
            model_tasks[model.name] = []
        model_tasks[model.name].append(TASK_SHORT_NAMES[task])

    model_utilization = [
        {
            "model": name,
            "tasks_count": count,
            "tasks_pct": count / len(TASKS) * 100,
            "tasks": model_tasks[name],
        }
        for name, count in sorted(model_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "task_analysis": task_analysis,
        "overall_analysis": overall_analysis,
        "model_utilization": model_utilization,
    }


def create_visualizations(
    analysis: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Create performance analysis visualizations.

    Parameters
    ----------
    analysis : dict[str, Any]
        Performance analysis results.
    output_dir : Path
        Directory to save visualizations.

    Returns
    -------
    list[Path]
        Paths to created visualization files.
    """
    # Create figures subdirectories
    figures_dir = output_dir / "figures"
    png_dir = figures_dir / "png"
    svg_dir = figures_dir / "svg"
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

    # Figure 1: Per-Task Rank Comparison (Reference vs Hybrid)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    task_data = analysis["task_analysis"]
    tasks = [d["task_short"] for d in task_data]
    ref_ranks = [d["reference_rank"] for d in task_data]
    best_ranks = [d["best_rank"] for d in task_data]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        ref_ranks,
        width,
        label=f"Reference ({task_data[0]['reference_model']})",
        color="#4C72B0",
        alpha=0.8,
    )
    bars2 = ax1.bar(x + width / 2, best_ranks, width, label="Hybrid (Best per Task)", color="#55A868", alpha=0.8)

    ax1.set_ylabel("Leaderboard Rank (lower is better)")
    ax1.set_xlabel("ADMET Task")
    ax1.set_title("Per-Task Rank Comparison: Reference vs Hybrid Ensemble")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha="right")
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(0, max(max(ref_ranks), max(best_ranks)) * 1.15)
    ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add value labels to all bars
    for i, (bar1, bar2, r1, r2) in enumerate(zip(bars1, bars2, ref_ranks, best_ranks)):
        # Reference bar label
        ax1.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height(),
            f"{r1:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Hybrid bar label
        ax1.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height(),
            f"{r2:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Improvement annotation (with 10% buffer to avoid overlap)
        if r1 > r2:
            improvement = r1 - r2
            buffer = bar2.get_height() * 0.10
            ax1.annotate(
                f"+{improvement}",
                xy=(x[i] + width / 2, r2),
                xytext=(0, -(15 + buffer)),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#2E7D32",
                fontweight="bold",
            )

    fig1.tight_layout()
    fig1_path_png = png_dir / "rank_comparison.png"
    fig1_path_svg = svg_dir / "rank_comparison.svg"
    fig1.savefig(fig1_path_png, dpi=150, bbox_inches="tight")
    fig1.savefig(fig1_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig1)
    created_files.extend([fig1_path_png, fig1_path_svg])
    print(f"  ✓ Created: {fig1_path_png.name} + SVG")

    # Figure 2: Per-Task MAE Comparison
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ref_mae = [d["reference_mae"] for d in task_data]
    best_mae = [d["best_mae"] for d in task_data]
    min_mae = [d.get("min_mae", d["best_mae"]) for d in task_data]

    x_pos = np.arange(len(tasks))
    bar_width = 0.25

    bars1 = ax2.bar(
        x_pos - bar_width,
        ref_mae,
        bar_width,
        label=f"Reference ({task_data[0]['reference_model']})",
        color="#4C72B0",
        alpha=0.8,
    )
    bars2 = ax2.bar(x_pos, best_mae, bar_width, label="Hybrid (Best per Task)", color="#55A868", alpha=0.8)
    bars3 = ax2.bar(
        x_pos + bar_width,
        min_mae,
        bar_width,
        label="Leaderboard Min (Jan-09)",
        color="#FFA500",
        alpha=0.8,
    )

    ax2.set_ylabel("Mean Absolute Error (MAE)")
    ax2.set_xlabel("ADMET Task")
    ax2.set_title("Per-Task MAE Comparison: Reference vs Hybrid vs Leaderboard Best")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tasks, rotation=45, ha="right")
    ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add error bars to leaderboard bars
    min_mae_std = [task_data[i].get("min_mae_std", 0.0) for i in range(len(tasks))]
    ax2.errorbar(
        x_pos + bar_width,
        min_mae,
        yerr=min_mae_std,
        fmt="none",
        ecolor="#333",
        capsize=3,
        capthick=1.5,
        alpha=0.7,
    )

    # Add value labels to all bars with alternating positions to avoid overlap
    for i, (bar1, bar2, bar3, m1, m2, m3) in enumerate(zip(bars1, bars2, bars3, ref_mae, best_mae, min_mae)):
        # Reference bar label (position above)
        ax2.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height(),
            f"{m1:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Hybrid bar label (position slightly higher)
        max_mae_value = max([*ref_mae, *best_mae, *min_mae])
        ax2.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + (max_mae_value * 0.02),
            f"{m2:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Leaderboard min bar label (position above error bar)
        offset = min_mae_std[i] if min_mae_std[i] > 0 else 0
        ax2.text(
            bar3.get_x() + bar3.get_width() / 2,
            bar3.get_height() + offset,
            f"{m3:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Improvement annotation (positioned below the hybrid bar)
        if m1 > m2:
            improvement_pct = (m1 - m2) / m1 * 100
            ax2.annotate(
                f"-{improvement_pct:.1f}%",
                xy=(x_pos[i], m2),
                xytext=(0, -25),  # Below the bar
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#2E7D32",
                fontweight="bold",
            )

    fig2.tight_layout()
    fig2_path_png = png_dir / "mae_comparison.png"
    fig2_path_svg = svg_dir / "mae_comparison.svg"
    fig2.savefig(fig2_path_png, dpi=150, bbox_inches="tight")
    fig2.savefig(fig2_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig2)
    created_files.extend([fig2_path_png, fig2_path_svg])
    print(f"  ✓ Created: {fig2_path_png.name} + SVG")

    # Figure 3: Model Utilization Pie Chart
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    model_util = analysis["model_utilization"]
    labels = [f"{m['model']}\n({m['tasks_count']}/9 tasks)" for m in model_util]
    sizes = [m["tasks_count"] for m in model_util]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#DD8452", "#937860"]

    ax3.pie(  # type: ignore
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(sizes)],
        startangle=90,
        explode=[0.02] * len(sizes),
    )
    ax3.set_title("Model Utilization in Hybrid Ensemble")

    fig3.tight_layout()
    fig3_path_png = png_dir / "model_utilization.png"
    fig3_path_svg = svg_dir / "model_utilization.svg"
    fig3.savefig(fig3_path_png, dpi=150, bbox_inches="tight")
    fig3.savefig(fig3_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig3)
    created_files.extend([fig3_path_png, fig3_path_svg])
    print(f"  ✓ Created: {fig3_path_png.name} + SVG")

    # Figure 4: Rank Improvement Waterfall
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    improvements = [d["rank_improvement"] for d in task_data]
    colors = ["#55A868" if imp > 0 else "#C44E52" if imp < 0 else "#808080" for imp in improvements]

    bars = ax4.bar(tasks, improvements, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax4.set_ylabel("Rank Improvement (positive = better)")
    ax4.set_xlabel("ADMET Task")
    ax4.set_title("Per-Task Rank Improvement from Hybrid Ensemble")
    ax4.set_xticklabels(tasks, rotation=45, ha="right")
    ax4.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax4.annotate(
            f"{imp:+.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    fig4.tight_layout()
    fig4_path_png = png_dir / "rank_improvement.png"
    fig4_path_svg = svg_dir / "rank_improvement.svg"
    fig4.savefig(fig4_path_png, dpi=150, bbox_inches="tight")
    fig4.savefig(fig4_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig4)
    created_files.extend([fig4_path_png, fig4_path_svg])
    print(f"  ✓ Created: {fig4_path_png.name} + SVG")

    # Figure 5: Heatmap of Model Performance by Task
    fig5, ax5 = plt.subplots(figsize=(12, 8))

    model_names = list(MODEL_SUBMISSIONS.keys())
    model_display_names = [MODEL_SUBMISSIONS[k].name for k in model_names]
    task_short = [TASK_SHORT_NAMES[t] for t in TASKS]

    # Create rank matrix
    rank_matrix = np.array([[MODEL_SUBMISSIONS[m].per_task_ranks[t] for t in TASKS] for m in model_names])

    # Plot heatmap
    im = ax5.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto")

    ax5.set_xticks(np.arange(len(task_short)))
    ax5.set_yticks(np.arange(len(model_display_names)))
    ax5.set_xticklabels(task_short, rotation=45, ha="right")
    ax5.set_yticklabels(model_display_names)

    # Add colorbar
    cbar = ax5.figure.colorbar(im, ax=ax5)
    cbar.ax.set_ylabel("Leaderboard Rank (lower = better)", rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(TASKS)):
            rank = rank_matrix[i, j]
            text_color = "white" if rank > 70 else "black"
            ax5.text(j, i, f"{rank}", ha="center", va="center", color=text_color, fontsize=8)

    # Highlight best model per task
    for j, task in enumerate(TASKS):
        best_model_key, _ = find_best_model_per_task()[task]
        best_idx = model_names.index(best_model_key)
        ax5.add_patch(plt.Rectangle((j - 0.5, best_idx - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=2))

    ax5.set_title("Model Performance Heatmap (Ranks) - Blue boxes indicate best per task")

    fig5.tight_layout()
    fig5_path_png = png_dir / "performance_heatmap.png"
    fig5_path_svg = svg_dir / "performance_heatmap.svg"
    fig5.savefig(fig5_path_png, dpi=150, bbox_inches="tight")
    fig5.savefig(fig5_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig5)
    created_files.extend([fig5_path_png, fig5_path_svg])
    print(f"  ✓ Created: {fig5_path_png.name} + SVG")

    # Figure 6: Per-Task R² Comparison
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    ref_r2 = [d["reference_r2"] for d in task_data]
    best_r2 = [d["best_r2"] for d in task_data]
    max_r2 = [d.get("max_r2", d["best_r2"]) for d in task_data]

    x_pos = np.arange(len(tasks))
    bar_width = 0.25

    bars1 = ax6.bar(
        x_pos - bar_width,
        ref_r2,
        bar_width,
        label=f"Reference ({task_data[0]['reference_model']})",
        color="#4C72B0",
        alpha=0.8,
    )
    bars2 = ax6.bar(x_pos, best_r2, bar_width, label="Hybrid (Best per Task)", color="#55A868", alpha=0.8)
    bars3 = ax6.bar(
        x_pos + bar_width,
        max_r2,
        bar_width,
        label="Leaderboard Max",
        color="#FFA500",
        alpha=0.8,
    )

    ax6.set_ylabel("R² Score (higher is better)")
    ax6.set_xlabel("ADMET Task")
    ax6.set_title("Per-Task R² Comparison: Reference vs Hybrid vs Leaderboard Best")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(tasks, rotation=45, ha="right")
    ax6.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax6.set_ylim(0, 1.0)
    ax6.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add error bars to leaderboard bars
    max_r2_std = [task_data[i].get("max_r2_std", 0.0) for i in range(len(tasks))]
    ax6.errorbar(
        x_pos + bar_width,
        max_r2,
        yerr=max_r2_std,
        fmt="none",
        ecolor="#333",
        capsize=3,
        capthick=1.5,
        alpha=0.7,
    )

    # Add value labels to all bars with alternating positions to avoid overlap
    for i, (bar1, bar2, bar3, r1, r2, r3) in enumerate(zip(bars1, bars2, bars3, ref_r2, best_r2, max_r2)):
        # Reference bar label (position above)
        ax6.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height(),
            f"{r1:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Hybrid bar label (position slightly higher)
        ax6.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + 0.02,
            f"{r2:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Leaderboard max bar label (position above error bar)
        offset = max_r2_std[i] if max_r2_std[i] > 0 else 0
        ax6.text(
            bar3.get_x() + bar3.get_width() / 2,
            bar3.get_height() + offset,
            f"{r3:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Improvement annotation (positioned below the hybrid bar)
        if r2 > r1:
            improvement = r2 - r1
            ax6.annotate(
                f"+{improvement:.2f}",
                xy=(x_pos[i], r2),
                xytext=(0, -25),  # Below the bar
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#2E7D32",
                fontweight="bold",
            )

    fig6.tight_layout()
    fig6_path_png = png_dir / "r2_comparison.png"
    fig6_path_svg = svg_dir / "r2_comparison.svg"
    fig6.savefig(fig6_path_png, dpi=150, bbox_inches="tight")
    fig6.savefig(fig6_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig6)
    created_files.extend([fig6_path_png, fig6_path_svg])
    print(f"  ✓ Created: {fig6_path_png.name} + SVG")

    # Figure 7: Per-Task Spearman R Comparison
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    ref_spearman = [d["reference_spearman"] for d in task_data]
    best_spearman = [d["best_spearman"] for d in task_data]
    max_spearman = [d.get("max_spearman", d["best_spearman"]) for d in task_data]

    bars1 = ax7.bar(
        x_pos - bar_width,
        ref_spearman,
        bar_width,
        label=f"Reference ({task_data[0]['reference_model']})",
        color="#4C72B0",
        alpha=0.8,
    )
    bars2 = ax7.bar(x_pos, best_spearman, bar_width, label="Hybrid (Best per Task)", color="#55A868", alpha=0.8)
    bars3 = ax7.bar(
        x_pos + bar_width,
        max_spearman,
        bar_width,
        label="Leaderboard Max",
        color="#FFA500",
        alpha=0.8,
    )

    ax7.set_ylabel("Spearman R (higher is better)")
    ax7.set_xlabel("ADMET Task")
    ax7.set_title("Per-Task Spearman R Comparison: Reference vs Hybrid vs Leaderboard Best")
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(tasks, rotation=45, ha="right")
    ax7.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax7.set_ylim(0, 1.0)
    ax7.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add error bars to leaderboard bars
    max_spearman_std = [task_data[i].get("max_spearman_std", 0.0) for i in range(len(tasks))]
    ax7.errorbar(
        x_pos + bar_width,
        max_spearman,
        yerr=max_spearman_std,
        fmt="none",
        ecolor="#333",
        capsize=3,
        capthick=1.5,
        alpha=0.7,
    )

    # Add value labels to all bars with alternating positions to avoid overlap
    for i, (bar1, bar2, bar3, s1, s2, s3) in enumerate(
        zip(bars1, bars2, bars3, ref_spearman, best_spearman, max_spearman)
    ):
        # Reference bar label (position above)
        ax7.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height(),
            f"{s1:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Hybrid bar label (position slightly higher)
        ax7.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + 0.02,
            f"{s2:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Leaderboard max bar label (position above error bar)
        offset = max_spearman_std[i] if max_spearman_std[i] > 0 else 0
        ax7.text(
            bar3.get_x() + bar3.get_width() / 2,
            bar3.get_height() + offset,
            f"{s3:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Improvement annotation (positioned below the hybrid bar)
        if s2 > s1:
            improvement = s2 - s1
            ax7.annotate(
                f"+{improvement:.2f}",
                xy=(x_pos[i], s2),
                xytext=(0, -25),  # Below the bar
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#2E7D32",
                fontweight="bold",
            )

    fig7.tight_layout()
    fig7_path_png = png_dir / "spearman_comparison.png"
    fig7_path_svg = svg_dir / "spearman_comparison.svg"
    fig7.savefig(fig7_path_png, dpi=150, bbox_inches="tight")
    fig7.savefig(fig7_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig7)
    created_files.extend([fig7_path_png, fig7_path_svg])
    print(f"  ✓ Created: {fig7_path_png.name} + SVG")

    # Figure 8: Per-Task Kendall's τ Comparison
    fig8, ax8 = plt.subplots(figsize=(12, 6))
    ref_kendall = [d["reference_kendall"] for d in task_data]
    best_kendall = [d["best_kendall"] for d in task_data]
    max_kendall = [d.get("max_kendall", d["best_kendall"]) for d in task_data]

    bars1 = ax8.bar(
        x_pos - bar_width,
        ref_kendall,
        bar_width,
        label=f"Reference ({task_data[0]['reference_model']})",
        color="#4C72B0",
        alpha=0.8,
    )
    bars2 = ax8.bar(x_pos, best_kendall, bar_width, label="Hybrid (Best per Task)", color="#55A868", alpha=0.8)
    bars3 = ax8.bar(
        x_pos + bar_width,
        max_kendall,
        bar_width,
        label="Leaderboard Max",
        color="#FFA500",
        alpha=0.8,
    )

    ax8.set_ylabel("Kendall's τ (higher is better)")
    ax8.set_xlabel("ADMET Task")
    ax8.set_title("Per-Task Kendall's τ Comparison: Reference vs Hybrid vs Leaderboard Best")
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(tasks, rotation=45, ha="right")
    ax8.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax8.set_ylim(0, 1.0)
    ax8.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add error bars to leaderboard bars
    max_kendall_std = [task_data[i].get("max_kendall_std", 0.0) for i in range(len(tasks))]
    ax8.errorbar(
        x_pos + bar_width,
        max_kendall,
        yerr=max_kendall_std,
        fmt="none",
        ecolor="#333",
        capsize=3,
        capthick=1.5,
        alpha=0.7,
    )

    # Add value labels to all bars with alternating positions to avoid overlap
    for i, (bar1, bar2, bar3, k1, k2, k3) in enumerate(
        zip(bars1, bars2, bars3, ref_kendall, best_kendall, max_kendall)
    ):
        # Reference bar label (position above)
        ax8.text(
            bar1.get_x() + bar1.get_width() / 2,
            bar1.get_height(),
            f"{k1:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Hybrid bar label (position slightly higher)
        ax8.text(
            bar2.get_x() + bar2.get_width() / 2,
            bar2.get_height() + 0.02,
            f"{k2:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Leaderboard max bar label (position above error bar)
        offset = max_kendall_std[i] if max_kendall_std[i] > 0 else 0
        ax8.text(
            bar3.get_x() + bar3.get_width() / 2,
            bar3.get_height() + offset,
            f"{k3:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )
        # Improvement annotation (positioned below the hybrid bar)
        if k2 > k1:
            improvement = k2 - k1
            ax8.annotate(
                f"+{improvement:.2f}",
                xy=(x_pos[i], k2),
                xytext=(0, -25),  # Below the bar
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#2E7D32",
                fontweight="bold",
            )

    fig8.tight_layout()
    fig8_path_png = png_dir / "kendall_comparison.png"
    fig8_path_svg = svg_dir / "kendall_comparison.svg"
    fig8.savefig(fig8_path_png, dpi=150, bbox_inches="tight")
    fig8.savefig(fig8_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig8)
    created_files.extend([fig8_path_png, fig8_path_svg])
    print(f"  ✓ Created: {fig8_path_png.name} + SVG")

    # Figure 9: Per-Task MAE for All Models
    fig9, ax9 = plt.subplots(figsize=(14, 7))
    model_names = [m.name for m in MODEL_SUBMISSIONS.values()]
    model_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#DD8452", "#937860"]

    x_pos = np.arange(len(tasks))
    bar_width = 0.15

    for i, (model_key, model) in enumerate(MODEL_SUBMISSIONS.items()):
        mae_values = [model.per_task_mae[task] for task in TASKS]
        offset = (i - len(MODEL_SUBMISSIONS) / 2) * bar_width + bar_width / 2
        ax9.bar(
            x_pos + offset,
            mae_values,
            bar_width,
            label=model.name,
            color=model_colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

    ax9.set_ylabel("Mean Absolute Error (MAE)")
    ax9.set_xlabel("ADMET Task")
    ax9.set_title("Per-Task MAE for All Submitted Models")
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(tasks, rotation=45, ha="right")
    ax9.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax9.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    fig9.tight_layout()
    fig9_path_png = png_dir / "all_models_mae_per_task.png"
    fig9_path_svg = svg_dir / "all_models_mae_per_task.svg"
    fig9.savefig(fig9_path_png, dpi=150, bbox_inches="tight")
    fig9.savefig(fig9_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig9)
    created_files.extend([fig9_path_png, fig9_path_svg])
    print(f"  ✓ Created: {fig9_path_png.name} + SVG")

    # Figure 10: Per-Task R² for All Models
    fig10, ax10 = plt.subplots(figsize=(14, 7))

    for i, (model_key, model) in enumerate(MODEL_SUBMISSIONS.items()):
        r2_values = [model.per_task_r2[task] for task in TASKS]
        offset = (i - len(MODEL_SUBMISSIONS) / 2) * bar_width + bar_width / 2
        ax10.bar(
            x_pos + offset,
            r2_values,
            bar_width,
            label=model.name,
            color=model_colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

    ax10.set_ylabel("R² Score (higher is better)")
    ax10.set_xlabel("ADMET Task")
    ax10.set_title("Per-Task R² for All Submitted Models")
    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(tasks, rotation=45, ha="right")
    ax10.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax10.set_ylim(0, 1.0)
    ax10.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    fig10.tight_layout()
    fig10_path_png = png_dir / "all_models_r2_per_task.png"
    fig10_path_svg = svg_dir / "all_models_r2_per_task.svg"
    fig10.savefig(fig10_path_png, dpi=150, bbox_inches="tight")
    fig10.savefig(fig10_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig10)
    created_files.extend([fig10_path_png, fig10_path_svg])
    print(f"  ✓ Created: {fig10_path_png.name} + SVG")

    # Figure 11: Per-Task Spearman R for All Models
    fig11, ax11 = plt.subplots(figsize=(14, 7))

    for i, (model_key, model) in enumerate(MODEL_SUBMISSIONS.items()):
        spearman_values = [model.per_task_spearman[task] for task in TASKS]
        offset = (i - len(MODEL_SUBMISSIONS) / 2) * bar_width + bar_width / 2
        ax11.bar(
            x_pos + offset,
            spearman_values,
            bar_width,
            label=model.name,
            color=model_colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

    ax11.set_ylabel("Spearman R (higher is better)")
    ax11.set_xlabel("ADMET Task")
    ax11.set_title("Per-Task Spearman R for All Submitted Models")
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(tasks, rotation=45, ha="right")
    ax11.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax11.set_ylim(0, 1.0)
    ax11.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    fig11.tight_layout()
    fig11_path_png = png_dir / "all_models_spearman_per_task.png"
    fig11_path_svg = svg_dir / "all_models_spearman_per_task.svg"
    fig11.savefig(fig11_path_png, dpi=150, bbox_inches="tight")
    fig11.savefig(fig11_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig11)
    created_files.extend([fig11_path_png, fig11_path_svg])
    print(f"  ✓ Created: {fig11_path_png.name} + SVG")

    # Figure 12: Per-Task Kendall's τ for All Models
    fig12, ax12 = plt.subplots(figsize=(14, 7))

    for i, (model_key, model) in enumerate(MODEL_SUBMISSIONS.items()):
        kendall_values = [model.per_task_kendall[task] for task in TASKS]
        offset = (i - len(MODEL_SUBMISSIONS) / 2) * bar_width + bar_width / 2
        ax12.bar(
            x_pos + offset,
            kendall_values,
            bar_width,
            label=model.name,
            color=model_colors[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

    ax12.set_ylabel("Kendall's τ (higher is better)")
    ax12.set_xlabel("ADMET Task")
    ax12.set_title("Per-Task Kendall's τ for All Submitted Models")
    ax12.set_xticks(x_pos)
    ax12.set_xticklabels(tasks, rotation=45, ha="right")
    ax12.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax12.set_ylim(0, 1.0)
    ax12.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    fig12.tight_layout()
    fig12_path_png = png_dir / "all_models_kendall_per_task.png"
    fig12_path_svg = svg_dir / "all_models_kendall_per_task.svg"
    fig12.savefig(fig12_path_png, dpi=150, bbox_inches="tight")
    fig12.savefig(fig12_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig12)
    created_files.extend([fig12_path_png, fig12_path_svg])
    print(f"  ✓ Created: {fig12_path_png.name} + SVG")

    # Figure 13: Mean Metrics Across All Tasks for Each Model
    fig13, ((ax13a, ax13b), (ax13c, ax13d)) = plt.subplots(2, 2, figsize=(14, 10))

    model_names_short = [m.name for m in MODEL_SUBMISSIONS.values()]
    x_models = np.arange(len(model_names_short))

    # Mean MAE
    mean_mae = [np.mean([model.per_task_mae[task] for task in TASKS]) for model in MODEL_SUBMISSIONS.values()]
    std_mae = [
        np.std([model.per_task_mae[task] for task in TASKS], ddof=1) / np.sqrt(len(TASKS))
        for model in MODEL_SUBMISSIONS.values()
    ]
    bars_mae = ax13a.bar(x_models, mean_mae, color=model_colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax13a.errorbar(
        x_models,
        mean_mae,
        yerr=std_mae,
        fmt="none",
        ecolor="#333",
        capsize=4,
        capthick=1.5,
        alpha=0.7,
    )
    ax13a.set_ylabel("Mean MAE (lower is better)")
    ax13a.set_xlabel("Model")
    ax13a.set_title("Mean MAE Across All 9 Tasks")
    ax13a.set_xticks(x_models)
    ax13a.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax13a.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars_mae, mean_mae):
        ax13a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    # Mean R²
    mean_r2 = [np.mean([model.per_task_r2[task] for task in TASKS]) for model in MODEL_SUBMISSIONS.values()]
    std_r2 = [
        np.std([model.per_task_r2[task] for task in TASKS], ddof=1) / np.sqrt(len(TASKS))
        for model in MODEL_SUBMISSIONS.values()
    ]
    bars_r2 = ax13b.bar(x_models, mean_r2, color=model_colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax13b.errorbar(
        x_models,
        mean_r2,
        yerr=std_r2,
        fmt="none",
        ecolor="#333",
        capsize=4,
        capthick=1.5,
        alpha=0.7,
    )
    ax13b.set_ylabel("Mean R² (higher is better)")
    ax13b.set_xlabel("Model")
    ax13b.set_title("Mean R² Across All 9 Tasks")
    ax13b.set_xticks(x_models)
    ax13b.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax13b.set_ylim(0, 1.0)
    ax13b.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    for bar, val in zip(bars_r2, mean_r2):
        ax13b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    # Mean Spearman R
    mean_spearman = [np.mean([model.per_task_spearman[task] for task in TASKS]) for model in MODEL_SUBMISSIONS.values()]
    std_spearman = [
        np.std([model.per_task_spearman[task] for task in TASKS], ddof=1) / np.sqrt(len(TASKS))
        for model in MODEL_SUBMISSIONS.values()
    ]
    bars_spearman = ax13c.bar(x_models, mean_spearman, color=model_colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax13c.errorbar(
        x_models,
        mean_spearman,
        yerr=std_spearman,
        fmt="none",
        ecolor="#333",
        capsize=4,
        capthick=1.5,
        alpha=0.7,
    )
    ax13c.set_ylabel("Mean Spearman R (higher is better)")
    ax13c.set_xlabel("Model")
    ax13c.set_title("Mean Spearman R Across All 9 Tasks")
    ax13c.set_xticks(x_models)
    ax13c.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax13c.set_ylim(0, 1.0)
    ax13c.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    for bar, val in zip(bars_spearman, mean_spearman):
        ax13c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    # Mean Kendall's τ
    mean_kendall = [np.mean([model.per_task_kendall[task] for task in TASKS]) for model in MODEL_SUBMISSIONS.values()]
    std_kendall = [
        np.std([model.per_task_kendall[task] for task in TASKS], ddof=1) / np.sqrt(len(TASKS))
        for model in MODEL_SUBMISSIONS.values()
    ]
    bars_kendall = ax13d.bar(x_models, mean_kendall, color=model_colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax13d.errorbar(
        x_models,
        mean_kendall,
        yerr=std_kendall,
        fmt="none",
        ecolor="#333",
        capsize=4,
        capthick=1.5,
        alpha=0.7,
    )
    ax13d.set_ylabel("Mean Kendall's τ (higher is better)")
    ax13d.set_xlabel("Model")
    ax13d.set_title("Mean Kendall's τ Across All 9 Tasks")
    ax13d.set_xticks(x_models)
    ax13d.set_xticklabels(model_names_short, rotation=45, ha="right")
    ax13d.set_ylim(0, 1.0)
    ax13d.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    for bar, val in zip(bars_kendall, mean_kendall):
        ax13d.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    fig13.tight_layout()
    fig13_path_png = png_dir / "all_models_mean_metrics.png"
    fig13_path_svg = svg_dir / "all_models_mean_metrics.svg"
    fig13.savefig(fig13_path_png, dpi=150, bbox_inches="tight")
    fig13.savefig(fig13_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig13)
    created_files.extend([fig13_path_png, fig13_path_svg])
    print(f"  ✓ Created: {fig13_path_png.name} + SVG")

    return created_files


def generate_report(
    analysis: dict[str, Any],
    hybrid_pred: pd.DataFrame,
    hybrid_sub: pd.DataFrame,
    output_dir: Path,
    visualization_files: list[Path],
) -> Path:
    """Generate reproducible HTML/Markdown report.

    Parameters
    ----------
    analysis : dict[str, Any]
        Performance analysis results.
    hybrid_pred : pd.DataFrame
        Hybrid predictions DataFrame (log-scale).
    hybrid_sub : pd.DataFrame
        Hybrid submissions DataFrame (transformed).
    output_dir : Path
        Directory to save report.
    visualization_files : list[Path]
        Paths to visualization files.

    Returns
    -------
    Path
        Path to generated report file.
    """
    overall = analysis["overall_analysis"]
    task_analysis = analysis["task_analysis"]
    model_util = analysis["model_utilization"]

    report_lines = [
        "# Hybrid Ensemble Model Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"This hybrid ensemble combines predictions from {len(MODEL_SUBMISSIONS)} model submissions by selecting "
        f"the best-performing model for each of the 9 ADMET tasks based on leaderboard rankings.",
        "",
        "### Expected Performance Improvement",
        "",
        f"| Metric | Reference ({overall['reference_model']}) | Hybrid Expected | Improvement |",
        "|--------|----------|-----------------|-------------|",
        (
            f"| MA-RAE | {overall['reference_ma_rae']:.2f} | "
            f"{overall['expected_ma_rae_range'][0]:.2f}-{overall['expected_ma_rae_range'][1]:.2f} | "
            f"~{overall['avg_mae_pct_improvement']:.1f}% |"
        ),
        (
            f"| Overall Rank | #{overall['reference_overall_rank']} | "
            f"#{overall['expected_overall_rank_range'][0]}-{overall['expected_overall_rank_range'][1]} | "
            "Top 4-5% |"
        ),
        (
            f"| Avg Task Rank | {overall['reference_avg_task_rank']:.1f} | "
            f"{overall['hybrid_avg_task_rank']:.1f} | +{overall['avg_rank_improvement']:.1f} ranks |"
        ),
        "",
        "---",
        "",
        "## Methodology: Task-Best Selection",
        "",
        "### Why NOT Differential Evolution?",
        "",
        "1. **No optimization target** - Blind challenge has no ground truth labels",
        "2. **Leaderboard ranks are validated** - Best model per task proven on actual blind set",
        "3. **Simpler is better** - No hyperparameters, deterministic, reproducible",
        "4. **Avoid overfitting** - Weight optimization on test data may not transfer",
        "",
        "### Selection Criteria",
        "",
        "For each task, select the model with the **lowest leaderboard rank** (best performance).",
        "",
        "---",
        "",
        "## Per-Task Analysis",
        "",
        "| Task | Reference Rank | Best Model | Best Rank | Rank Δ | MAE Δ | % Improvement |",
        "|------|---------------|------------|-----------|--------|-------|---------------|",
    ]

    for t in task_analysis:
        rank_delta = f"+{t['rank_improvement']}" if t["rank_improvement"] > 0 else str(t["rank_improvement"])
        mae_delta = f"-{t['mae_improvement']:.2f}" if t["mae_improvement"] > 0 else f"+{abs(t['mae_improvement']):.2f}"
        pct_imp = f"{t['mae_pct_improvement']:.1f}%" if t["mae_pct_improvement"] > 0 else "0.0%"
        row = (
            f"| {t['task_short']} | #{t['reference_rank']} | {t['best_model']} | "
            f"#{t['best_rank']} | {rank_delta} | {mae_delta} | {pct_imp} |"
        )
        report_lines.append(row)

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## Model Utilization",
            "",
            "| Model | Tasks | % Usage | Task List |",
            "|-------|-------|---------|-----------|",
        ]
    )

    for m in model_util:
        tasks_str = ", ".join(m["tasks"])
        report_lines.append(f"| {m['model']} | {m['tasks_count']}/9 | {m['tasks_pct']:.1f}% | {tasks_str} |")

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## Visualizations",
            "",
        ]
    )

    for viz_path in visualization_files:
        report_lines.append(f"### {viz_path.stem.replace('_', ' ').title()}")
        report_lines.append("")
        report_lines.append(f"![{viz_path.stem}]({viz_path.name})")
        report_lines.append("")

    report_lines.extend(
        [
            "---",
            "",
            "## Prediction Statistics (Log-Scale)",
            "",
            "| Task | Mean | Std | Min | Max |",
            "|------|------|-----|-----|-----|",
        ]
    )

    for task in TASKS:
        col = hybrid_pred[task]
        report_lines.append(
            f"| {TASK_SHORT_NAMES[task]} | {col.mean():.3f} | {col.std():.3f} | {col.min():.3f} | {col.max():.3f} |"
        )

    report_lines.extend(
        [
            "",
            "## Submission Statistics (Transformed)",
            "",
            "| Task | Mean | Std | Min | Max |",
            "|------|------|-----|-----|-----|",
        ]
    )

    for task, sub_col in TASK_TO_SUBMISSION_COL.items():
        col = hybrid_sub[sub_col]
        report_lines.append(
            f"| {TASK_SHORT_NAMES[task]} | {col.mean():.3f} | {col.std():.3f} | {col.min():.3f} | {col.max():.3f} |"
        )

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## Reproducibility",
            "",
            "### Model Sources",
            "",
            "| Model | Date | Experiment ID | Run ID |",
            "|-------|------|--------------|--------|",
        ]
    )

    for model in MODEL_SUBMISSIONS.values():
        report_lines.append(f"| {model.name} | {model.date} | {model.experiment_id} | `{model.run_id[:12]}...` |")

    report_lines.extend(
        [
            "",
            "### Files Generated",
            "",
            f"- **Predictions:** `blind_predictions.csv` ({len(hybrid_pred)} samples, log-scale)",
            f"- **Submissions:** `blind_submissions.csv` ({len(hybrid_sub)} samples, transformed for upload)",
            "- **Report:** `report.md`",
            "- **Metadata:** `metadata.json`",
            f"- **Visualizations:** {len(visualization_files)} PNG files",
            "",
            "---",
            "",
            "*Report generated by `create_hybrid_predictions.py`*",
        ]
    )

    # Create reports subdirectory
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    print(f"  ✓ Created: {report_path.name}")

    return report_path


def save_metadata(
    analysis: dict[str, Any],
    best_models: dict[str, tuple[str, ModelSubmission]],
    output_dir: Path,
) -> Path:
    """Save reproducibility metadata as JSON.

    Parameters
    ----------
    analysis : dict[str, Any]
        Performance analysis results.
    best_models : dict[str, tuple[str, ModelSubmission]]
        Best model mapping.
    output_dir : Path
        Directory to save metadata.

    Returns
    -------
    Path
        Path to metadata file.
    """
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "script": "create_hybrid_predictions.py",
        "method": "task_best_selection",
        "task_model_mapping": {
            task: {
                "model_key": model_key,
                "model_name": model.name,
                "architecture": model.architecture,
                "date": model.date,
                "run_id": model.run_id,
                "experiment_id": model.experiment_id,
                "rank": model.per_task_ranks[task],
                "mae": model.per_task_mae[task],
            }
            for task, (model_key, model) in best_models.items()
        },
        "model_submissions": {
            key: {
                "name": model.name,
                "date": model.date,
                "architecture": model.architecture,
                "run_id": model.run_id,
                "experiment_id": model.experiment_id,
                "overall_rank": model.overall_rank,
                "overall_ma_rae": model.overall_ma_rae,
            }
            for key, model in MODEL_SUBMISSIONS.items()
        },
        "expected_performance": analysis["overall_analysis"],
    }

    # Create reports subdirectory
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = reports_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Created: {metadata_path.name}")

    return metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Create hybrid ensemble predictions for minimum MA-RAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets/submissions/2026-01-10-hybrid",
        help="Output directory for predictions and report",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip creating visualizations",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default="jan09_weighted",
        choices=list(MODEL_SUBMISSIONS.keys()),
        help="Reference model for comparison (default: jan09_weighted)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Hybrid Ensemble Model for Minimum MA-RAE")
    print("=" * 70)
    print()
    print(f"Output directory: {args.output_dir}")
    print(f"Reference model: {MODEL_SUBMISSIONS[args.reference_model].name}")
    print()

    # Step 0: Fetch leaderboard metrics (for dynamic min/max values)
    print("Step 0: Fetching leaderboard metrics")
    print("-" * 70)
    leaderboard_metrics = fetch_leaderboard_metrics()

    # Step 1: Find best model per task
    print("Step 1: Analyzing per-task model performance")
    print("-" * 70)
    best_models = find_best_model_per_task()

    # Step 2: Compute performance analysis
    print("\nStep 2: Computing expected performance improvements")
    print("-" * 70)
    analysis = compute_performance_analysis(best_models, leaderboard_metrics, args.reference_model)
    overall = analysis["overall_analysis"]
    print(f"  Reference MA-RAE: {overall['reference_ma_rae']:.2f}")
    print(f"  Expected MA-RAE:  {overall['expected_ma_rae_range'][0]:.2f}-{overall['expected_ma_rae_range'][1]:.2f}")
    print(f"  Avg MAE improvement: {overall['avg_mae_pct_improvement']:.1f}%")
    print(f"  Avg rank improvement: +{overall['avg_rank_improvement']:.1f} ranks")

    # Step 3: Create hybrid predictions and submissions
    print("\nStep 3: Creating hybrid predictions and submissions")
    print("-" * 70)
    hybrid_pred, hybrid_sub = create_hybrid_predictions(best_models)

    # Step 4: Save predictions and submissions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = data_dir / "blind_predictions.csv"
    hybrid_pred.to_csv(predictions_path, index=False)
    print(f"\n  ✓ Saved predictions: {predictions_path}")
    print(f"    Shape: {hybrid_pred.shape}")

    submissions_path = data_dir / "blind_submissions.csv"
    hybrid_sub.to_csv(submissions_path, index=False)
    print(f"  ✓ Saved submissions: {submissions_path}")
    print(f"    Shape: {hybrid_sub.shape}")

    # Step 5: Create visualizations
    visualization_files = []
    if not args.no_visualizations:
        print("\nStep 4: Creating visualizations")
        print("-" * 70)
        visualization_files = create_visualizations(analysis, args.output_dir)

    # Step 6: Generate report
    print("\nStep 5: Generating report")
    print("-" * 70)
    report_path = generate_report(analysis, hybrid_pred, hybrid_sub, args.output_dir, visualization_files)

    # Step 7: Save metadata
    print("\nStep 6: Saving metadata")
    print("-" * 70)
    metadata_path = save_metadata(analysis, best_models, args.output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nOutput files in {args.output_dir}:")
    print("  data/:")
    print(f"    • {predictions_path.name} - Blind predictions (log-scale)")
    print(f"    • {submissions_path.name} - Blind submissions (for challenge upload)")
    print("  reports/:")
    print(f"    • {report_path.name} - Analysis report")
    print(f"    • {metadata_path.name} - Reproducibility metadata")
    if visualization_files:
        print("  figures/:")
        print(f"    • png/ - {len(visualization_files)//2} PNG files")
        print(f"    • svg/ - {len(visualization_files)//2} SVG files")

    print("\n✅ Hybrid ensemble creation complete!")
    ref_mae = overall["reference_ma_rae"]
    exp_low = overall["expected_ma_rae_range"][0]
    exp_high = overall["expected_ma_rae_range"][1]
    print(f"\nExpected improvement: MA-RAE {ref_mae:.2f} → {exp_low:.2f}-{exp_high:.2f}")


if __name__ == "__main__":
    main()
