#!/usr/bin/env python3
"""Generate top-100 Chemprop ensemble configs from MLflow HPO results with visualizations.

This script queries MLflow experiments for Chemprop HPO runs, sorts by val_loss,
generates individual YAML config files for ensemble training, and creates
comprehensive visualizations for HPO analysis.

Usage:
    python scripts/hpo/generate_chemprop_ensemble_configs_from_mlflow.py
    python scripts/hpo/generate_chemprop_ensemble_configs_from_mlflow.py --skip-plots
    python scripts/hpo/generate_chemprop_ensemble_configs_from_mlflow.py --experiment-ids "17,14"
    python scripts/hpo/generate_chemprop_ensemble_configs_from_mlflow.py --dry-run

The script will:
1. Query MLflow experiments (default: 17, 14, 13, 3) for Chemprop HPO runs
2. Sort by val_loss (ascending) and take top 100
3. Generate configs/2-hpo-ensemble/2_chemprop_v2/ensemble_chemprop_hpo_001.yaml through _100.yaml
4. Generate comprehensive visualizations in configs/2-hpo-ensemble/2_chemprop_v2/plots/

Visualizations include:
- Parameter distributions (all trials vs top-k)
- Correlation heatmaps
- Parallel coordinates plots
- Learning rate schedule analysis
- FFN architecture comparisons
- Config similarity clustering (PCA)
- Performance vs hyperparameter scatter plots
"""

from __future__ import annotations

import argparse
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Constants
# =============================================================================

ENSEMBLE_DATA_DIR = "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data"

TARGET_COLS = [
    "LogD",
    "Log KSOL",
    "Log HLM CLint",
    "Log MLM CLint",
    "Log Caco-2 Permeability Papp A>B",
    "Log Caco-2 Permeability Efflux",
    "Log MPPB",
    "Log MBPB",
    "Log MGMB",
]

# FFN type mapping from HPO to config format
FFN_TYPE_MAPPING = {
    "mlp": "regression",
    "moe": "mixture_of_experts",
    "branched": "branched",
    "regression": "regression",
    "mixture_of_experts": "mixture_of_experts",
}

# Default parameter values when missing
PARAM_DEFAULTS = {
    "depth": 3,
    "message_hidden_dim": 700,
    "ffn_num_layers": 4,
    "ffn_hidden_dim": 200,
    "dropout": 0.15,
    "batch_size": 128,
    "batch_norm": True,
    "ffn_type": "regression",
    "weight_decay": 0.0,
    "learning_rate": 0.001,
    "lr_warmup_ratio": 0.1,
    "lr_final_ratio": 0.1,
    "aggregation": "norm",
    "n_experts": 4,
    "trunk_depth": 2,
    "trunk_hidden_dim": 500,
    "joint_sampling_enabled": True,
    "joint_sampling_alpha": 0.02,
}


# =============================================================================
# MLflow Query Functions
# =============================================================================


def query_mlflow_experiments(
    tracking_uri: str,
    experiment_ids: list[str],
    metric_name: str = "val_loss",
) -> pd.DataFrame:
    """Query multiple MLflow experiments and merge results into a DataFrame.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking server URI.
    experiment_ids : list[str]
        List of experiment IDs to query.
    metric_name : str
        Name of the metric to extract (default: val_loss).

    Returns
    -------
    pd.DataFrame
        DataFrame with all runs, their parameters, and metrics.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    all_runs = []
    valid_experiment_ids = []

    # Validate experiments
    print(f"Validating {len(experiment_ids)} experiments...")
    for exp_id in experiment_ids:
        try:
            exp = client.get_experiment(exp_id)
            if exp.lifecycle_stage == "active":
                valid_experiment_ids.append(exp_id)
                print(f"  Found: {exp_id} ({exp.name})")
            else:
                print(f"  Skipping deleted experiment: {exp_id}")
        except Exception as e:
            print(f"  Warning: Could not find experiment {exp_id}: {e}")

    if not valid_experiment_ids:
        raise ValueError("No valid experiments found")

    # Query runs from each experiment
    print(f"\nQuerying runs from {len(valid_experiment_ids)} experiments...")
    for exp_id in valid_experiment_ids:
        exp = client.get_experiment(exp_id)
        runs = client.search_runs(
            experiment_ids=[exp_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=10000,
        )

        for run in runs:
            # Skip runs without the target metric
            if metric_name not in run.data.metrics:
                continue

            run_data = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "experiment_name": exp.name,
                "val_loss": run.data.metrics.get(metric_name),
                "val_mae": run.data.metrics.get("val_mae"),
                "val_rmse": run.data.metrics.get("val_rmse"),
            }

            # Add all parameters
            for key, value in run.data.params.items():
                run_data[key] = value

            all_runs.append(run_data)

        print(f"  {exp.name}: {len([r for r in runs if metric_name in r.data.metrics])} runs with {metric_name}")

    if not all_runs:
        raise ValueError(f"No runs found with metric '{metric_name}'")

    df = pd.DataFrame(all_runs)
    print(f"\nTotal runs collected: {len(df)}")
    return df


# =============================================================================
# Parameter Extraction Functions
# =============================================================================


def safe_get_param(
    row: pd.Series,
    param_names: list[str],
    default: Any,
    convert_type: type | None = None,
) -> Any:
    """Safely extract a parameter from multiple possible column names.

    Parameters
    ----------
    row : pd.Series
        DataFrame row.
    param_names : list[str]
        List of possible parameter names to try.
    default : Any
        Default value if not found.
    convert_type : type, optional
        Type to convert the value to.

    Returns
    -------
    Any
        The parameter value or default.
    """
    for name in param_names:
        if name in row.index:
            value = row[name]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                if pd.notna(value):
                    if convert_type is not None:
                        try:
                            return convert_type(value)
                        except (ValueError, TypeError):
                            continue
                    return value
    return default


def parse_bool(value: Any) -> bool:
    """Parse a boolean value from various formats."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def extract_chemprop_hyperparams(row: pd.Series) -> dict:
    """Extract hyperparameters from a DataFrame row (MLflow run data).

    Parameters
    ----------
    row : pd.Series
        A row from the MLflow runs DataFrame.

    Returns
    -------
    dict
        Dictionary of extracted hyperparameters.
    """
    # Model architecture parameters
    # Check non-prefixed names first (more common), then prefixed
    depth = safe_get_param(row, ["depth", "config/depth"], PARAM_DEFAULTS["depth"], int)
    message_hidden_dim = safe_get_param(
        row,
        ["message_hidden_dim", "config/message_hidden_dim"],
        PARAM_DEFAULTS["message_hidden_dim"],
        int,
    )
    aggregation = safe_get_param(
        row, ["aggregation", "config/aggregation"], PARAM_DEFAULTS["aggregation"], str
    )

    # FFN parameters
    ffn_type_raw = safe_get_param(
        row, ["ffn_type", "config/ffn_type"], PARAM_DEFAULTS["ffn_type"], str
    )
    ffn_type = FFN_TYPE_MAPPING.get(ffn_type_raw, "regression")
    ffn_num_layers = safe_get_param(
        row,
        ["ffn_num_layers", "config/ffn_num_layers", "num_layers", "config/num_layers"],
        PARAM_DEFAULTS["ffn_num_layers"],
        int,
    )
    ffn_hidden_dim = safe_get_param(
        row,
        ["ffn_hidden_dim", "config/ffn_hidden_dim", "hidden_dim", "config/hidden_dim"],
        PARAM_DEFAULTS["ffn_hidden_dim"],
        int,
    )
    dropout = safe_get_param(
        row, ["dropout", "config/dropout"], PARAM_DEFAULTS["dropout"], float
    )
    batch_norm = parse_bool(
        safe_get_param(row, ["batch_norm", "config/batch_norm"], PARAM_DEFAULTS["batch_norm"])
    )

    # Conditional FFN parameters
    n_experts = safe_get_param(row, ["n_experts", "config/n_experts"], None, int)
    trunk_depth = safe_get_param(row, ["trunk_depth", "config/trunk_depth"], None, int)
    trunk_hidden_dim = safe_get_param(
        row, ["trunk_hidden_dim", "config/trunk_hidden_dim"], None, int
    )

    # Learning rate schedule
    learning_rate = safe_get_param(
        row, ["learning_rate", "config/learning_rate"], PARAM_DEFAULTS["learning_rate"], float
    )
    lr_warmup_ratio = safe_get_param(
        row, ["lr_warmup_ratio", "config/lr_warmup_ratio"], PARAM_DEFAULTS["lr_warmup_ratio"], float
    )
    lr_final_ratio = safe_get_param(
        row, ["lr_final_ratio", "config/lr_final_ratio"], PARAM_DEFAULTS["lr_final_ratio"], float
    )

    # Calculate init_lr, max_lr, final_lr
    max_lr = learning_rate
    init_lr = learning_rate * lr_warmup_ratio
    final_lr = learning_rate * lr_final_ratio

    # Training parameters
    batch_size = safe_get_param(
        row, ["batch_size", "config/batch_size"], PARAM_DEFAULTS["batch_size"], int
    )
    weight_decay = safe_get_param(
        row, ["weight_decay", "config/weight_decay"], PARAM_DEFAULTS["weight_decay"], float
    )

    # Joint sampling parameters
    joint_sampling_enabled = parse_bool(
        safe_get_param(
            row,
            ["joint_sampling_enabled", "config/joint_sampling_enabled"],
            PARAM_DEFAULTS["joint_sampling_enabled"],
        )
    )
    joint_sampling_alpha = safe_get_param(
        row,
        ["joint_sampling_alpha", "config/joint_sampling_alpha",
         "task_sampling_alpha", "config/task_sampling_alpha"],
        PARAM_DEFAULTS["joint_sampling_alpha"],
        float,
    )

    return {
        # Model architecture
        "depth": depth,
        "message_hidden_dim": message_hidden_dim,
        "aggregation": aggregation,
        "ffn_type": ffn_type,
        "ffn_num_layers": ffn_num_layers,
        "ffn_hidden_dim": ffn_hidden_dim,
        "dropout": dropout,
        "batch_norm": batch_norm,
        # Conditional FFN params
        "n_experts": n_experts,
        "trunk_depth": trunk_depth,
        "trunk_hidden_dim": trunk_hidden_dim,
        # Learning rate
        "init_lr": init_lr,
        "max_lr": max_lr,
        "final_lr": final_lr,
        # Training
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        # Joint sampling
        "joint_sampling_enabled": joint_sampling_enabled,
        "joint_sampling_alpha": joint_sampling_alpha,
        # Metrics and provenance
        "val_loss": row.get("val_loss"),
        "val_mae": row.get("val_mae"),
        "run_id": row.get("run_id"),
        "experiment_id": row.get("experiment_id"),
        "experiment_name": row.get("experiment_name"),
    }


# =============================================================================
# Config Generation Functions
# =============================================================================


def generate_yaml_header(
    rank: int,
    experiment_name: str,
    experiment_id: str,
    run_id: str,
    val_loss: float,
) -> str:
    """Generate the YAML file header with provenance information.

    Parameters
    ----------
    rank : int
        Rank of this configuration (1-based).
    experiment_name : str
        Name of the source MLflow experiment.
    experiment_id : str
        ID of the source MLflow experiment.
    run_id : str
        ID of the source MLflow run.
    val_loss : float
        Validation loss from the run.

    Returns
    -------
    str
        YAML header comment block.
    """
    return f"""# Ensemble configuration file generated from Chemprop single-fold model results.
# This file is auto-generated. Do not modify manually.
# Original single-fold:
#   Experiment Name: {experiment_name}
#   Experiment ID: {experiment_id}
#   Run ID: {run_id}
#   Val Loss: {val_loss:.6f}
# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


def build_chemprop_ensemble_config(params: dict, rank: int) -> dict:
    """Build a complete ensemble config dictionary for Chemprop.

    Parameters
    ----------
    params : dict
        Extracted hyperparameters.
    rank : int
        Rank of this configuration (1-based).

    Returns
    -------
    dict
        Complete configuration dictionary.
    """
    # Build chemprop model config
    chemprop_config = {
        "depth": params["depth"],
        "message_hidden_dim": params["message_hidden_dim"],
        "aggregation": params["aggregation"],
        "ffn_type": params["ffn_type"],
        "num_layers": params["ffn_num_layers"],
        "hidden_dim": params["ffn_hidden_dim"],
        "dropout": round(params["dropout"], 6),
        "batch_norm": params["batch_norm"],
    }

    # Add conditional FFN-type-specific parameters
    if params["ffn_type"] == "mixture_of_experts" and params.get("n_experts") is not None:
        chemprop_config["n_experts"] = params["n_experts"]
    elif params["ffn_type"] == "branched":
        if params.get("trunk_depth") is not None:
            chemprop_config["trunk_n_layers"] = params["trunk_depth"]
        else:
            chemprop_config["trunk_n_layers"] = PARAM_DEFAULTS["trunk_depth"]
        if params.get("trunk_hidden_dim") is not None:
            chemprop_config["trunk_hidden_dim"] = params["trunk_hidden_dim"]
        else:
            chemprop_config["trunk_hidden_dim"] = PARAM_DEFAULTS["trunk_hidden_dim"]

    config = {
        "data": {
            "data_dir": ENSEMBLE_DATA_DIR,
            "splits": None,
            "folds": None,
            "test_file": "assets/dataset/set/local_test.csv",
            "blind_file": "assets/dataset/set/blind_test.csv",
            "output_dir": None,
            "smiles_col": "SMILES",
            "target_cols": TARGET_COLS,
            "target_weights": [1.0] * len(TARGET_COLS),
        },
        "model": {
            "type": "chemprop",
            "chemprop": chemprop_config,
        },
        "optimization": {
            "criterion": "MAE",
            "init_lr": round(params["init_lr"], 8),
            "max_lr": round(params["max_lr"], 8),
            "final_lr": round(params["final_lr"], 8),
            "warmup_epochs": 5,
            "max_epochs": 150,
            "patience": 15,
            "batch_size": params["batch_size"],
            "num_workers": 4,
            "accumulate_grad_batches": 1,
            "seed": 42,
            "weight_decay": round(params["weight_decay"], 8),
            "progress_bar": False,
        },
        "performance_optimization": {
            "use_mixed_precision": True,
        },
        "mlflow": {
            "tracking": True,
            "tracking_uri": "http://127.0.0.1:8084",
            "experiment_name": "chemprop_hpo_ensemble_v2",
            "run_name": f"rank_{rank:03d}",
            "nested": True,
        },
        "post_training": {
            "generate_plots": True,
            "plot_dpi": 150,
            "plot_formats": ["png"],
            "cache_predictions": True,
            "compute_test_metrics": True,
            "compute_train_metrics": False,
        },
        "inter_task_affinity": {
            "enabled": False,
            "compute_every_n_steps": 2,
            "log_every_n_steps": 10,
            "log_epoch_summary": True,
            "log_step_matrices": False,
            "lookahead_lr": 0.001,
            "use_optimizer_lr": True,
            "exclude_param_patterns": ["predictor", "ffn", "output", "head", "readout"],
            "log_to_mlflow": True,
        },
        "joint_sampling": {
            "enabled": params["joint_sampling_enabled"],
            "task_oversampling": {
                "alpha": round(params["joint_sampling_alpha"], 6),
            },
            "curriculum": {
                "enabled": False,
                "quality_col": "Quality",
                "qualities": ["high", "medium", "low"],
                "patience": 5,
                "strategy": "sampled",
                "reset_early_stopping_on_phase_change": False,
                "log_per_quality_metrics": True,
                "seed": 42,
            },
            "num_samples": None,
            "seed": 42,
            "increment_seed_per_epoch": True,
            "log_to_mlflow": True,
        },
        "ray": {
            "max_parallel": 6,
            "num_cpus": 24,
            "num_gpus": 2,
            "gpu_ids": [0, 1],
        },
        "logging": {
            "enabled": True,
            "verbose": 0,
            "max_total_logs_gb": 1.0,
            "fail_on_upload_error": True,
        },
    }

    return config


def represent_none(dumper: yaml.Dumper, _: None) -> yaml.Node:
    """Represent None as null in YAML."""
    return dumper.represent_scalar("tag:yaml.org,2002:null", "null")


def represent_float(dumper: yaml.Dumper, value: float) -> yaml.Node:
    """Represent floats with appropriate precision."""
    if value == int(value):
        return dumper.represent_int(int(value))
    # Use scientific notation for very small numbers
    if abs(value) < 1e-4 and value != 0:
        return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.2e}")
    return dumper.represent_float(value)


def save_config(config: dict, header: str, output_path: Path) -> None:
    """Save config to YAML file with header and proper formatting.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    header : str
        Header comment to prepend.
    output_path : Path
        Output file path.
    """
    yaml.add_representer(type(None), represent_none)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)


# =============================================================================
# Visualization Functions
# =============================================================================


class ChempropHPOVisualizer:
    """Comprehensive HPO visualization generator adapted for Chemprop parameters."""

    # Chemprop-specific numeric parameters to analyze
    # Support both with and without config/ prefix
    NUMERIC_PARAMS = [
        "learning_rate",
        "lr_warmup_ratio",
        "lr_final_ratio",
        "dropout",
        "weight_decay",
        "depth",
        "message_hidden_dim",
        "ffn_num_layers",
        "ffn_hidden_dim",
        "batch_size",
        "n_experts",
        "trunk_depth",
        "trunk_hidden_dim",
        "joint_sampling_alpha",
        # Also check config/ prefixed versions
        "config/learning_rate",
        "config/lr_warmup_ratio",
        "config/lr_final_ratio",
        "config/dropout",
        "config/weight_decay",
        "config/depth",
        "config/message_hidden_dim",
        "config/ffn_num_layers",
        "config/ffn_hidden_dim",
        "config/batch_size",
        "config/n_experts",
        "config/trunk_depth",
        "config/trunk_hidden_dim",
        "config/joint_sampling_alpha",
    ]

    CATEGORICAL_PARAMS = [
        "ffn_type",
        "batch_norm",
        "joint_sampling_enabled",
        "aggregation",
        "config/ffn_type",
        "config/batch_norm",
        "config/joint_sampling_enabled",
        "config/aggregation",
    ]

    # Display names for cleaner plots
    PARAM_DISPLAY_NAMES = {
        # Non-prefixed versions
        "learning_rate": "Learning Rate",
        "lr_warmup_ratio": "LR Warmup Ratio",
        "lr_final_ratio": "LR Final Ratio",
        "dropout": "Dropout",
        "weight_decay": "Weight Decay",
        "depth": "MPNN Depth",
        "message_hidden_dim": "Message Hidden Dim",
        "ffn_type": "FFN Type",
        "ffn_num_layers": "FFN Layers",
        "ffn_hidden_dim": "FFN Hidden Dim",
        "batch_size": "Batch Size",
        "batch_norm": "Batch Norm",
        "n_experts": "N Experts (MoE)",
        "trunk_depth": "Trunk Depth (Branched)",
        "trunk_hidden_dim": "Trunk Hidden Dim (Branched)",
        "aggregation": "Aggregation",
        "joint_sampling_alpha": "Joint Sampling Alpha",
        "joint_sampling_enabled": "Joint Sampling",
        # Config-prefixed versions
        "config/learning_rate": "Learning Rate",
        "config/lr_warmup_ratio": "LR Warmup Ratio",
        "config/lr_final_ratio": "LR Final Ratio",
        "config/dropout": "Dropout",
        "config/weight_decay": "Weight Decay",
        "config/depth": "MPNN Depth",
        "config/message_hidden_dim": "Message Hidden Dim",
        "config/ffn_type": "FFN Type",
        "config/ffn_num_layers": "FFN Layers",
        "config/ffn_hidden_dim": "FFN Hidden Dim",
        "config/batch_size": "Batch Size",
        "config/batch_norm": "Batch Norm",
        "config/n_experts": "N Experts (MoE)",
        "config/trunk_depth": "Trunk Depth (Branched)",
        "config/trunk_hidden_dim": "Trunk Hidden Dim (Branched)",
        "config/aggregation": "Aggregation",
        "config/joint_sampling_alpha": "Joint Sampling Alpha",
        "config/joint_sampling_enabled": "Joint Sampling",
        "val_loss": "Validation Loss",
    }

    def __init__(self, all_trials_df: pd.DataFrame, top_k_df: pd.DataFrame, output_dir: Path):
        """Initialize the visualizer.

        Parameters
        ----------
        all_trials_df : pd.DataFrame
            DataFrame with all HPO trials.
        top_k_df : pd.DataFrame
            DataFrame with top-k trials.
        output_dir : Path
            Output directory for plots.
        """
        self.all_trials = all_trials_df.copy()
        self.top_k = top_k_df.copy()
        self.output_dir = output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert string columns to numeric where possible
        for col in self.NUMERIC_PARAMS:
            if col in self.all_trials.columns:
                self.all_trials[col] = pd.to_numeric(self.all_trials[col], errors="coerce")
            if col in self.top_k.columns:
                self.top_k[col] = pd.to_numeric(self.top_k[col], errors="coerce")

        # Add rank column to top_k
        self.top_k["rank"] = range(1, len(self.top_k) + 1)

        # Style settings
        self.figsize_single = (10, 6)
        self.figsize_wide = (14, 6)
        self.figsize_tall = (10, 12)
        self.figsize_large = (16, 12)
        self.dpi = 150

    def _get_display_name(self, col: str) -> str:
        """Get human-readable display name for a column."""
        return self.PARAM_DISPLAY_NAMES.get(col, col.replace("config/", "").replace("_", " ").title())

    def _save_figure(self, fig, name: str) -> Path:
        """Save figure to output directory."""
        import matplotlib.pyplot as plt

        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return path

    def generate_all_plots(self) -> list[Path]:
        """Generate all visualizations and return list of saved paths."""
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")

        saved_paths = []
        print("\nGenerating visualizations...")

        # 1. Parameter distributions comparison
        print("  [1/10] Parameter distributions (all vs top-k)...")
        saved_paths.append(self.plot_parameter_distributions())

        # 2. Correlation heatmap
        print("  [2/10] Correlation heatmap...")
        saved_paths.append(self.plot_correlation_heatmap())

        # 3. Parallel coordinates
        print("  [3/10] Parallel coordinates plot...")
        saved_paths.append(self.plot_parallel_coordinates())

        # 4. FFN type comparison
        print("  [4/10] FFN type performance comparison...")
        saved_paths.append(self.plot_ffn_type_comparison())

        # 5. Learning rate analysis
        print("  [5/10] Learning rate schedule analysis...")
        saved_paths.append(self.plot_learning_rate_analysis())

        # 6. Top configs heatmap
        print("  [6/10] Top configurations heatmap...")
        saved_paths.append(self.plot_top_configs_heatmap())

        # 7. Parameter importance scatter plots
        print("  [7/10] Parameter vs performance scatter plots...")
        saved_paths.append(self.plot_parameter_performance_scatter())

        # 8. Config similarity (PCA)
        print("  [8/10] Configuration similarity (PCA)...")
        saved_paths.append(self.plot_config_similarity_pca())

        # 9. Hyperparameter search space coverage
        print("  [9/10] Search space coverage...")
        saved_paths.append(self.plot_search_space_coverage())

        # 10. Summary statistics table
        print("  [10/10] Summary statistics...")
        saved_paths.append(self.plot_summary_statistics())

        print(f"\n  Saved {len(saved_paths)} visualization files to: {self.output_dir}/")
        return saved_paths

    def plot_parameter_distributions(self) -> Path:
        """Plot distributions of each parameter comparing all trials vs top-k."""
        import matplotlib.pyplot as plt

        # Filter to params that exist in the data
        numeric_cols = [c for c in self.NUMERIC_PARAMS if c in self.all_trials.columns]
        n_params = len(numeric_cols)
        if n_params == 0:
            # Create empty plot
            fig, ax = plt.subplots(figsize=self.figsize_single)
            ax.text(
                0.5, 0.5, "No numeric parameters found",
                ha="center", va="center", transform=ax.transAxes,
            )
            return self._save_figure(fig, "01_parameter_distributions")

        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]

            # Get valid data
            all_data = self.all_trials[col].dropna()
            top_data = self.top_k[col].dropna()

            if len(all_data) == 0:
                ax.set_visible(False)
                continue

            # Use log scale for certain parameters
            use_log = col in ["config/learning_rate", "config/weight_decay", "config/lr_final_ratio"]

            if use_log and (all_data > 0).all():
                all_data = np.log10(all_data)
                top_data = np.log10(top_data[top_data > 0]) if len(top_data) > 0 else top_data
                xlabel = f"log10({self._get_display_name(col)})"
            else:
                xlabel = self._get_display_name(col)

            # Plot histograms
            all_label = f"All ({len(self.all_trials)})"
            ax.hist(all_data, bins=30, alpha=0.5, label=all_label, color="steelblue", density=True)
            if len(top_data) > 0:
                top_label = f"Top {len(self.top_k)}"
                ax.hist(top_data, bins=20, alpha=0.7, label=top_label, color="coral", density=True)

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(self._get_display_name(col))

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            "Hyperparameter Distributions: All Trials vs Top-K",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        return self._save_figure(fig, "01_parameter_distributions")

    def plot_correlation_heatmap(self) -> Path:
        """Plot correlation heatmap between hyperparameters and val_loss."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Select numeric columns including val_loss
        cols = [c for c in self.NUMERIC_PARAMS + ["val_loss"] if c in self.top_k.columns]
        data = self.top_k[cols].dropna(axis=1, how="all")

        if len(data.columns) < 2:
            fig, ax = plt.subplots(figsize=self.figsize_single)
            ax.text(
                0.5, 0.5, "Not enough numeric parameters for correlation",
                ha="center", va="center", transform=ax.transAxes,
            )
            return self._save_figure(fig, "02_correlation_heatmap")

        # Calculate correlation matrix
        corr_matrix = data.corr()

        # Rename for display
        display_names = [self._get_display_name(c) for c in corr_matrix.columns]
        corr_matrix.columns = display_names
        corr_matrix.index = display_names

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 8},
        )

        ax.set_title(f"Hyperparameter Correlations (Top {len(self.top_k)} Configs)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return self._save_figure(fig, "02_correlation_heatmap")

    def plot_parallel_coordinates(self) -> Path:
        """Plot parallel coordinates for top configs."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Select key parameters for parallel coordinates
        key_params = [
            "config/learning_rate",
            "config/dropout",
            "config/depth",
            "config/message_hidden_dim",
            "config/ffn_num_layers",
            "config/ffn_hidden_dim",
            "config/batch_size",
        ]
        key_params = [c for c in key_params if c in self.top_k.columns]

        if len(key_params) < 2:
            fig, ax = plt.subplots(figsize=self.figsize_wide)
            ax.text(
                0.5, 0.5, "Not enough parameters for parallel coordinates",
                ha="center", va="center", transform=ax.transAxes,
            )
            return self._save_figure(fig, "03_parallel_coordinates")

        # Prepare data (normalize to 0-1 range for each parameter)
        plot_data = self.top_k[key_params + ["val_loss"]].copy()
        normalized_data = plot_data.copy()

        for col in key_params:
            col_min, col_max = plot_data[col].min(), plot_data[col].max()
            if col_max > col_min:
                normalized_data[col] = (plot_data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0.5

        fig, ax = plt.subplots(figsize=self.figsize_wide)

        # Color by val_loss (lower = better = more green)
        norm = Normalize(vmin=plot_data["val_loss"].min(), vmax=plot_data["val_loss"].max())
        cmap = plt.get_cmap("RdYlGn_r")

        # Plot each config as a line
        x = range(len(key_params))
        for i, (idx, row) in enumerate(normalized_data.iterrows()):
            color = cmap(norm(plot_data.iloc[i]["val_loss"]))
            alpha = 0.8 if i < 10 else 0.3
            linewidth = 2 if i < 10 else 0.5
            ax.plot(x, [row[c] for c in key_params], color=color, alpha=alpha, linewidth=linewidth)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([self._get_display_name(c) for c in key_params], rotation=45, ha="right")
        ax.set_ylabel("Normalized Value (0-1)")
        ax.set_title(f"Parallel Coordinates: Top {len(self.top_k)} Configurations", fontsize=14, fontweight="bold")

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Validation Loss")

        plt.tight_layout()
        return self._save_figure(fig, "03_parallel_coordinates")

    def plot_ffn_type_comparison(self) -> Path:
        """Compare performance across FFN types."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        ffn_col = "config/ffn_type"
        if ffn_col not in self.all_trials.columns or ffn_col not in self.top_k.columns:
            fig, ax = plt.subplots(figsize=self.figsize_single)
            ax.text(0.5, 0.5, "FFN type data not available", ha="center", va="center", transform=ax.transAxes)
            return self._save_figure(fig, "04_ffn_type_comparison")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Box plot of val_loss by FFN type (all trials)
        ax = axes[0]
        sns.boxplot(data=self.all_trials, x=ffn_col, y="val_loss", ax=ax, palette="Set2")
        ax.set_xlabel("FFN Type")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Val Loss Distribution by FFN Type\n(All Trials)")

        # 2. Count of top-k by FFN type
        ax = axes[1]
        ffn_counts = self.top_k[ffn_col].value_counts()
        colors = sns.color_palette("Set2", len(ffn_counts))
        bars = ax.bar(ffn_counts.index, ffn_counts.values, color=colors)
        ax.set_xlabel("FFN Type")
        ax.set_ylabel("Count")
        ax.set_title(f"FFN Type Distribution\n(Top {len(self.top_k)})")
        for bar, count in zip(bars, ffn_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", fontsize=10)

        # 3. Val loss vs rank by FFN type
        ax = axes[2]
        for ffn_type in self.top_k[ffn_col].unique():
            subset = self.top_k[self.top_k[ffn_col] == ffn_type]
            ax.scatter(subset["rank"], subset["val_loss"], label=ffn_type, alpha=0.7, s=50)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Val Loss vs Rank by FFN Type\n(Top {len(self.top_k)})")
        ax.legend()

        fig.suptitle("FFN Architecture Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "04_ffn_type_comparison")

    def plot_learning_rate_analysis(self) -> Path:
        """Analyze learning rate schedule parameters."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        lr_col = "config/learning_rate"
        warmup_col = "config/lr_warmup_ratio"
        final_col = "config/lr_final_ratio"

        # 1. Learning rate distribution
        ax = axes[0, 0]
        if lr_col in self.all_trials.columns:
            lr_all = pd.to_numeric(self.all_trials[lr_col], errors="coerce").dropna()
            lr_top = pd.to_numeric(self.top_k[lr_col], errors="coerce").dropna()
            if len(lr_all) > 0 and (lr_all > 0).all():
                ax.hist(np.log10(lr_all), bins=30, alpha=0.5, label="All", color="steelblue", density=True)
            if len(lr_top) > 0 and (lr_top > 0).all():
                ax.hist(
                    np.log10(lr_top), bins=20, alpha=0.7,
                    label=f"Top {len(self.top_k)}", color="coral", density=True,
                )
            ax.set_xlabel("log10(Learning Rate)")
            ax.set_ylabel("Density")
            ax.set_title("Learning Rate Distribution")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Learning rate data not available", ha="center", va="center", transform=ax.transAxes)

        # 2. LR vs val_loss scatter
        ax = axes[0, 1]
        if lr_col in self.top_k.columns:
            lr_data = pd.to_numeric(self.top_k[lr_col], errors="coerce")
            valid_mask = lr_data.notna() & (lr_data > 0)
            if valid_mask.sum() > 0:
                scatter = ax.scatter(
                    lr_data[valid_mask],
                    self.top_k.loc[valid_mask, "val_loss"],
                    c=self.top_k.loc[valid_mask, "rank"],
                    cmap="viridis_r",
                    alpha=0.7,
                    s=50,
                )
                ax.set_xscale("log")
                ax.set_xlabel("Learning Rate (log scale)")
                ax.set_ylabel("Validation Loss")
                ax.set_title(f"LR vs Val Loss (Top {len(self.top_k)})")
                plt.colorbar(scatter, ax=ax, label="Rank")
            else:
                ax.text(0.5, 0.5, "No valid LR data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Learning rate data not available", ha="center", va="center", transform=ax.transAxes)

        # 3. Warmup ratio vs final ratio
        ax = axes[1, 0]
        if warmup_col in self.top_k.columns and final_col in self.top_k.columns:
            warmup_data = pd.to_numeric(self.top_k[warmup_col], errors="coerce")
            final_data = pd.to_numeric(self.top_k[final_col], errors="coerce")
            valid_mask = warmup_data.notna() & final_data.notna() & (warmup_data > 0) & (final_data > 0)
            if valid_mask.sum() > 0:
                scatter = ax.scatter(
                    warmup_data[valid_mask],
                    final_data[valid_mask],
                    c=self.top_k.loc[valid_mask, "val_loss"],
                    cmap="RdYlGn_r",
                    alpha=0.7,
                    s=50,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("LR Warmup Ratio (log)")
                ax.set_ylabel("LR Final Ratio (log)")
                ax.set_title("LR Schedule Parameters")
                plt.colorbar(scatter, ax=ax, label="Val Loss")
            else:
                ax.text(0.5, 0.5, "No valid warmup/final ratio data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Warmup/final ratio data not available", ha="center", va="center", transform=ax.transAxes)

        # 4. Effective LR range (init to final)
        ax = axes[1, 1]
        if all(c in self.top_k.columns for c in [lr_col, warmup_col, final_col]):
            lr = pd.to_numeric(self.top_k[lr_col], errors="coerce")
            warmup = pd.to_numeric(self.top_k[warmup_col], errors="coerce")
            final = pd.to_numeric(self.top_k[final_col], errors="coerce")
            valid_mask = lr.notna() & warmup.notna() & final.notna() & (lr > 0) & (warmup > 0) & (final > 0)

            if valid_mask.sum() > 0:
                init_lr = lr * warmup
                final_lr = lr * final

                n_show = min(20, valid_mask.sum())
                for i in range(n_show):
                    idx = valid_mask[valid_mask].index[i]
                    ax.plot([0, 1, 2], [init_lr.loc[idx], lr.loc[idx], final_lr.loc[idx]], alpha=0.5, linewidth=1)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(["Init LR", "Max LR", "Final LR"])
                ax.set_yscale("log")
                ax.set_ylabel("Learning Rate (log)")
                ax.set_title(f"LR Schedules (Top {n_show})")
            else:
                ax.text(0.5, 0.5, "No valid LR schedule data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "LR schedule data not available", ha="center", va="center", transform=ax.transAxes)

        fig.suptitle("Learning Rate Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "05_learning_rate_analysis")

    def plot_top_configs_heatmap(self) -> Path:
        """Create a heatmap showing normalized hyperparameters for top configs."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Select top 25 for readability
        n_show = min(25, len(self.top_k))
        top_subset = self.top_k.head(n_show)

        # Select key parameters
        params = [
            "config/learning_rate",
            "config/dropout",
            "config/depth",
            "config/message_hidden_dim",
            "config/ffn_num_layers",
            "config/ffn_hidden_dim",
            "config/batch_size",
        ]
        params = [c for c in params if c in top_subset.columns]

        if len(params) < 2:
            fig, ax = plt.subplots(figsize=self.figsize_single)
            ax.text(0.5, 0.5, "Not enough parameters for heatmap", ha="center", va="center", transform=ax.transAxes)
            return self._save_figure(fig, "06_top_configs_heatmap")

        # Normalize each parameter to 0-1
        data = top_subset[params].copy()
        for col in params:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            col_min, col_max = data[col].min(), data[col].max()
            if col_max > col_min:
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 0.5

        # Rename columns for display
        data.columns = [self._get_display_name(c) for c in params]
        data.index = [f"Rank {i+1}" for i in range(n_show)]

        fig, ax = plt.subplots(figsize=(10, 12))
        sns.heatmap(
            data,
            annot=False,
            cmap="YlOrRd",
            cbar_kws={"label": "Normalized Value (0-1)"},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"Top {n_show} Configurations (Normalized Parameters)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Hyperparameter")
        ax.set_ylabel("Configuration Rank")

        plt.tight_layout()
        return self._save_figure(fig, "06_top_configs_heatmap")

    def plot_parameter_performance_scatter(self) -> Path:
        """Scatter plots of key parameters vs validation loss."""
        import matplotlib.pyplot as plt

        # Key parameters to analyze
        params = [
            ("config/learning_rate", True),
            ("config/dropout", False),
            ("config/depth", False),
            ("config/message_hidden_dim", False),
            ("config/ffn_num_layers", False),
            ("config/batch_size", False),
        ]
        params = [(p, log) for p, log in params if p in self.all_trials.columns]

        if len(params) == 0:
            fig, ax = plt.subplots(figsize=self.figsize_single)
            ax.text(
                0.5, 0.5, "No parameters available for scatter plots",
                ha="center", va="center", transform=ax.transAxes,
            )
            return self._save_figure(fig, "07_parameter_performance_scatter")

        n_params = len(params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]

        for idx, (param, use_log) in enumerate(params):
            ax = axes[idx]

            # Plot all trials in gray
            x_all = pd.to_numeric(self.all_trials[param], errors="coerce")
            y_all = self.all_trials["val_loss"]
            valid_all = x_all.notna() & y_all.notna()
            if use_log:
                valid_all = valid_all & (x_all > 0)
            ax.scatter(x_all[valid_all], y_all[valid_all], alpha=0.1, s=10, color="gray", label="All trials")

            # Highlight top-k
            x_top = pd.to_numeric(self.top_k[param], errors="coerce")
            y_top = self.top_k["val_loss"]
            valid_top = x_top.notna() & y_top.notna()
            if use_log:
                valid_top = valid_top & (x_top > 0)
            if valid_top.sum() > 0:
                top_label = f"Top {len(self.top_k)}"
                ax.scatter(
                    x_top[valid_top],
                    y_top[valid_top],
                    c=self.top_k.loc[valid_top, "rank"],
                    cmap="plasma_r",
                    alpha=0.8,
                    s=30,
                    label=top_label,
                )

            if use_log:
                ax.set_xscale("log")

            ax.set_xlabel(self._get_display_name(param))
            ax.set_ylabel("Validation Loss")
            ax.set_title(f"{self._get_display_name(param)} vs Val Loss")

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("Parameter vs Performance Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "07_parameter_performance_scatter")

    def plot_config_similarity_pca(self) -> Path:
        """PCA visualization of config similarity."""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Select numeric parameters for PCA
        params = [c for c in self.NUMERIC_PARAMS if c in self.top_k.columns]
        if len(params) < 2:
            fig, ax = plt.subplots(figsize=self.figsize_single)
            ax.text(0.5, 0.5, "Not enough parameters for PCA", ha="center", va="center", transform=ax.transAxes)
            return self._save_figure(fig, "08_config_similarity_pca")

        data = self.top_k[params].copy()
        for col in params:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.fillna(0)

        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. PCA colored by val_loss
        ax = axes[0]
        scatter = ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=self.top_k["val_loss"],
            cmap="RdYlGn_r",
            s=50,
            alpha=0.7,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title("PCA: Colored by Val Loss")
        plt.colorbar(scatter, ax=ax, label="Val Loss")

        # Annotate top 5
        for i in range(min(5, len(pca_result))):
            ax.annotate(f"#{i+1}", (pca_result[i, 0], pca_result[i, 1]), fontsize=8, fontweight="bold")

        # 2. PCA colored by FFN type
        ax = axes[1]
        ffn_col = "config/ffn_type"
        if ffn_col in self.top_k.columns:
            ffn_types = self.top_k[ffn_col].unique()
            cmap_set1 = plt.get_cmap("Set1")
            colors = cmap_set1(np.linspace(0, 1, len(ffn_types)))
            for ffn_type, color in zip(ffn_types, colors):
                mask = self.top_k[ffn_col] == ffn_type
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    c=[color],
                    label=ffn_type,
                    s=50,
                    alpha=0.7,
                )
            ax.legend()
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], c="steelblue", s=50, alpha=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title("PCA: Colored by FFN Type")

        title = f"Configuration Similarity Analysis (Top {len(self.top_k)})"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "08_config_similarity_pca")

    def plot_search_space_coverage(self) -> Path:
        """Visualize search space coverage."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        lr_col = "config/learning_rate"
        dropout_col = "config/dropout"
        hidden_col = "config/ffn_hidden_dim"
        layers_col = "config/ffn_num_layers"

        # 1. 2D histogram: learning_rate vs dropout
        ax = axes[0, 0]
        if lr_col in self.all_trials.columns and dropout_col in self.all_trials.columns:
            x = pd.to_numeric(self.all_trials[lr_col], errors="coerce")
            y = pd.to_numeric(self.all_trials[dropout_col], errors="coerce")
            valid = x.notna() & y.notna() & (x > 0)
            if valid.sum() > 0:
                ax.hist2d(np.log10(x[valid]), y[valid], bins=30, cmap="Blues")
                ax.set_xlabel("log10(Learning Rate)")
                ax.set_ylabel("Dropout")
                ax.set_title("Search Space: LR vs Dropout")
            else:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)

        # 2. 2D histogram: ffn_hidden_dim vs ffn_num_layers
        ax = axes[0, 1]
        if hidden_col in self.all_trials.columns and layers_col in self.all_trials.columns:
            x = pd.to_numeric(self.all_trials[hidden_col], errors="coerce")
            y = pd.to_numeric(self.all_trials[layers_col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() > 0:
                ax.hist2d(x[valid], y[valid], bins=[20, 5], cmap="Blues")
                ax.set_xlabel("FFN Hidden Dim")
                ax.set_ylabel("FFN Num Layers")
                ax.set_title("Search Space: FFN Architecture")
            else:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center", transform=ax.transAxes)

        # 3. Trial count by experiment
        ax = axes[1, 0]
        if "experiment_name" in self.all_trials.columns:
            exp_counts = self.all_trials["experiment_name"].value_counts().sort_index()
            ax.bar(range(len(exp_counts)), exp_counts.values, color="steelblue")
            ax.set_xticks(range(len(exp_counts)))
            labels = [s.replace("chemprop_hpo_", "") for s in exp_counts.index]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Number of Trials")
            ax.set_title("Trials per Experiment")
        else:
            ax.text(0.5, 0.5, "Experiment data not available", ha="center", va="center", transform=ax.transAxes)

        # 4. Val loss distribution
        ax = axes[1, 1]
        val_loss = self.all_trials["val_loss"].dropna()
        if len(val_loss) > 0:
            ax.hist(val_loss, bins=50, color="steelblue", alpha=0.7)
            ax.axvline(val_loss.quantile(0.1), color="coral", linestyle="--", label="10th percentile")
            ax.legend()
        ax.set_xlabel("Validation Loss")
        ax.set_ylabel("Count")
        ax.set_title("Val Loss Distribution")

        fig.suptitle("Search Space Coverage Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "09_search_space_coverage")

    def plot_summary_statistics(self) -> Path:
        """Create summary statistics table as a figure."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # 1. Summary statistics table for top-k
        ax = axes[0]
        ax.axis("off")

        params = [c for c in self.NUMERIC_PARAMS if c in self.top_k.columns]
        stats_data = []
        for param in params:
            data = pd.to_numeric(self.top_k[param], errors="coerce").dropna()
            if len(data) > 0:
                stats_data.append(
                    [
                        self._get_display_name(param),
                        f"{data.min():.6g}",
                        f"{data.max():.6g}",
                        f"{data.mean():.6g}",
                        f"{data.std():.6g}",
                        f"{data.median():.6g}",
                    ]
                )

        if stats_data:
            table = ax.table(
                cellText=stats_data,
                colLabels=["Parameter", "Min", "Max", "Mean", "Std", "Median"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        title = f"Summary Statistics (Top {len(self.top_k)} Configurations)"
        ax.set_title(title, fontsize=14, fontweight="bold", y=0.98)

        # 2. Top 10 configurations table
        ax = axes[1]
        ax.axis("off")

        top_10 = self.top_k.head(10)
        table_data = []

        ffn_col = "config/ffn_type"
        layers_col = "config/ffn_num_layers"
        hidden_col = "config/ffn_hidden_dim"
        batch_col = "config/batch_size"
        lr_col = "config/learning_rate"
        dropout_col = "config/dropout"

        for idx, row in top_10.iterrows():
            ffn_type = row.get(ffn_col, "N/A")
            ffn_layers = safe_get_param(row, [layers_col], "N/A", int)
            ffn_hidden = safe_get_param(row, [hidden_col], "N/A", int)
            batch_size = safe_get_param(row, [batch_col], "N/A", int)
            lr = safe_get_param(row, [lr_col], 0.0, float)
            dropout = safe_get_param(row, [dropout_col], 0.0, float)

            table_data.append(
                [
                    int(row["rank"]),
                    f"{row['val_loss']:.6f}",
                    str(ffn_type),
                    str(ffn_layers),
                    str(ffn_hidden),
                    str(batch_size),
                    f"{lr:.2e}" if isinstance(lr, float) else str(lr),
                    f"{dropout:.3f}" if isinstance(dropout, float) else str(dropout),
                ]
            )

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=["Rank", "Val Loss", "FFN Type", "FFN Layers", "FFN Hidden", "Batch Size", "LR", "Dropout"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        ax.set_title("Top 10 Configurations", fontsize=14, fontweight="bold", y=0.98)

        plt.tight_layout()
        return self._save_figure(fig, "10_summary_statistics")


# =============================================================================
# Main Function
# =============================================================================


def main() -> None:
    """Main function to generate Chemprop ensemble configs from MLflow."""
    parser = argparse.ArgumentParser(
        description="Generate Chemprop ensemble configs from MLflow HPO experiments"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:8084",
        help="MLflow tracking URI (default: http://127.0.0.1:8084)",
    )
    parser.add_argument(
        "--experiment-ids",
        type=str,
        default="17,14,13,3",
        help="Comma-separated MLflow experiment IDs (default: 17,14,13,3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/2-hpo-ensemble/2_chemprop_v2"),
        help="Output directory for configs and plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top configs to generate (default: 100)",
    )
    parser.add_argument(
        "--metric",
        default="val_loss",
        help="Metric to sort by - ascending (default: val_loss)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing files",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )
    args = parser.parse_args()

    experiment_ids = [e.strip() for e in args.experiment_ids.split(",")]

    print("=" * 80)
    print("Chemprop Ensemble Config Generator from MLflow")
    print("=" * 80)
    print(f"MLflow Tracking URI: {args.tracking_uri}")
    print(f"Experiment IDs: {experiment_ids}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Top K: {args.top_k}")
    print(f"Metric: {args.metric}")
    print("=" * 80)

    # Query MLflow experiments
    df = query_mlflow_experiments(
        tracking_uri=args.tracking_uri,
        experiment_ids=experiment_ids,
        metric_name=args.metric,
    )

    # Sort by metric and take top-k
    df = df.sort_values(args.metric).reset_index(drop=True)
    top_k_df = df.head(args.top_k).copy()

    print(f"\nTop {len(top_k_df)} configurations by {args.metric}:")
    print("-" * 80)
    for i, (_, row) in enumerate(top_k_df.head(10).iterrows()):
        rank = i + 1
        # Check both prefixed and non-prefixed column names
        ffn_type = row.get("ffn_type", row.get("config/ffn_type", "N/A"))
        batch_size = row.get("batch_size", row.get("config/batch_size", "N/A"))
        exp_name = row.get("experiment_name", "N/A")
        print(
            f"  Rank {rank:3d}: {args.metric}={row[args.metric]:.6f}, "
            f"ffn_type={ffn_type}, batch_size={batch_size}, exp={exp_name}"
        )
    if len(top_k_df) > 10:
        print(f"  ... and {len(top_k_df) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate config files
    print(f"\nGenerating {len(top_k_df)} config files in {args.output_dir}/")
    for i, (_, row) in enumerate(top_k_df.iterrows()):
        rank = i + 1
        params = extract_chemprop_hyperparams(row)
        config = build_chemprop_ensemble_config(params, rank)
        header = generate_yaml_header(
            rank=rank,
            experiment_name=params["experiment_name"],
            experiment_id=str(params["experiment_id"]),
            run_id=params["run_id"],
            val_loss=params["val_loss"],
        )

        output_file = args.output_dir / f"ensemble_chemprop_hpo_{rank:03d}.yaml"
        save_config(config, header, output_file)

        if rank <= 5 or rank == len(top_k_df):
            print(f"  Created: {output_file.name} (val_loss={params['val_loss']:.6f})")
        elif rank == 6:
            print("  ...")

    print(f"\nDone! Generated {len(top_k_df)} ensemble config files.")
    print(f"Output directory: {args.output_dir.resolve()}")

    # Generate visualizations
    if not args.skip_plots:
        print("\n" + "=" * 80)
        try:
            visualizer = ChempropHPOVisualizer(df, top_k_df, args.output_dir)
            visualizer.generate_all_plots()
        except ImportError as e:
            print(f"Warning: Could not generate plots due to missing dependencies: {e}")
            print("Install matplotlib, seaborn, and scikit-learn to generate plots.")
        print("=" * 80)


if __name__ == "__main__":
    main()
