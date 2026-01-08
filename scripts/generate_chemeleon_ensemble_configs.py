#!/usr/bin/env python3
"""Generate top-100 CheMeleon ensemble configs from HPO results with visualizations.

This script reads HPO result CSV files, merges them, sorts by val_loss,
generates individual YAML config files for ensemble training, and creates
comprehensive visualizations for HPO analysis.

Usage:
    python scripts/generate_chemeleon_ensemble_configs.py
    python scripts/generate_chemeleon_ensemble_configs.py --skip-plots  # Skip visualization

The script will:
1. Read all hpo_results_*.csv files from chemeleon-hpo/
2. Merge and deduplicate trials by trial_id
3. Sort by val_loss (ascending) and take top 100
4. Generate configs/3-hpo-ensemble-chemeleon/ensemble_chemeleon_hpo_001.yaml through _100.yaml
5. Generate comprehensive visualizations in configs/3-hpo-ensemble-chemeleon/plots/

Visualizations include:
- Parameter distributions (all trials vs top-k)
- Correlation heatmaps
- Parallel coordinates plots
- Learning rate schedule analysis
- FFN architecture comparisons
- Config similarity clustering (PCA/t-SNE)
- Performance vs hyperparameter scatter plots
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_and_merge_hpo_results(csv_dir: Path) -> pd.DataFrame:
    """Load all HPO result CSVs and merge into single DataFrame."""
    csv_files = sorted(csv_dir.glob("hpo_results_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No hpo_results_*.csv files found in {csv_dir}")

    print(f"Found {len(csv_files)} HPO result files:")
    for f in csv_files:
        print(f"  - {f.name}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Deduplicate by trial_id, keeping the row with lowest val_loss
    merged = merged.sort_values("val_loss").drop_duplicates(subset=["trial_id"], keep="first")

    print(f"\nTotal unique trials after deduplication: {len(merged)}")
    return merged


def extract_hyperparams(row: pd.Series) -> dict:
    """Extract hyperparameters from a DataFrame row."""
    # Learning rate schedule parameters
    learning_rate = row["config/learning_rate"]
    lr_warmup_ratio = row["config/lr_warmup_ratio"]
    lr_final_ratio = row["config/lr_final_ratio"]

    # Calculate init_lr, max_lr, final_lr from HPO parameters
    # HPO uses: learning_rate as base, warmup_ratio scales to max, final_ratio scales down
    max_lr = learning_rate
    init_lr = learning_rate / lr_warmup_ratio if lr_warmup_ratio > 0 else learning_rate * 0.1
    final_lr = learning_rate * lr_final_ratio

    # FFN architecture
    ffn_type = row["config/ffn_type"]
    ffn_num_layers = int(row["config/ffn_num_layers"])
    ffn_hidden_dim = int(row["config/ffn_hidden_dim"])

    # Optional FFN-type specific parameters
    n_experts = int(row["config/n_experts"]) if pd.notna(row.get("config/n_experts")) else None
    trunk_depth = int(row["config/trunk_depth"]) if pd.notna(row.get("config/trunk_depth")) else None
    trunk_hidden_dim = int(row["config/trunk_hidden_dim"]) if pd.notna(row.get("config/trunk_hidden_dim")) else None

    # Training parameters
    dropout = row["config/dropout"]
    weight_decay = row["config/weight_decay"]
    batch_size = int(row["config/batch_size"])
    batch_norm = bool(row["config/batch_norm"])

    # Encoder freezing
    freeze_encoder = bool(row["config/freeze_encoder"])
    unfreeze_encoder_epoch = (
        int(row["config/unfreeze_encoder_epoch"]) if pd.notna(row.get("config/unfreeze_encoder_epoch")) else None
    )
    unfreeze_encoder_lr_multiplier = row.get("config/unfreeze_encoder_lr_multiplier", 0.1)
    if pd.isna(unfreeze_encoder_lr_multiplier):
        unfreeze_encoder_lr_multiplier = 0.1

    # Joint sampling
    joint_sampling_enabled = bool(row.get("config/joint_sampling_enabled", False))
    joint_sampling_alpha = row.get("config/joint_sampling_alpha", 0.0)
    if pd.isna(joint_sampling_alpha):
        joint_sampling_alpha = 0.0

    return {
        "init_lr": init_lr,
        "max_lr": max_lr,
        "final_lr": final_lr,
        "ffn_type": ffn_type,
        "ffn_num_layers": ffn_num_layers,
        "ffn_hidden_dim": ffn_hidden_dim,
        "n_experts": n_experts,
        "trunk_depth": trunk_depth,
        "trunk_hidden_dim": trunk_hidden_dim,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "batch_norm": batch_norm,
        "freeze_encoder": freeze_encoder,
        "unfreeze_encoder_epoch": unfreeze_encoder_epoch,
        "unfreeze_encoder_lr_multiplier": unfreeze_encoder_lr_multiplier,
        "joint_sampling_enabled": joint_sampling_enabled,
        "joint_sampling_alpha": joint_sampling_alpha,
        "val_loss": row["val_loss"],
        "val_mae": row.get("val_mae", row["val_loss"]),
        "trial_id": row["trial_id"],
        "epoch": int(row["epoch"]),
    }


def build_ensemble_config(params: dict, rank: int) -> dict:
    """Build a complete ensemble config dictionary."""
    # Build chemeleon model config
    chemeleon_config = {
        "checkpoint_path": "auto",
        "freeze_encoder": params["freeze_encoder"],
        "ffn_type": params["ffn_type"],
        "ffn_hidden_dim": params["ffn_hidden_dim"],
        "ffn_num_layers": params["ffn_num_layers"],
        "dropout": round(params["dropout"], 6),
        "batch_norm": params["batch_norm"],
        "unfreeze_schedule": {
            "freeze_encoder": params["freeze_encoder"],
            "unfreeze_encoder_epoch": params["unfreeze_encoder_epoch"],
            "unfreeze_encoder_lr_multiplier": round(params["unfreeze_encoder_lr_multiplier"], 6),
        },
    }

    # Add FFN-type specific parameters
    if params["ffn_type"] == "mixture_of_experts" and params["n_experts"] is not None:
        chemeleon_config["n_experts"] = params["n_experts"]
    elif params["ffn_type"] == "branched":
        if params["trunk_depth"] is not None:
            chemeleon_config["trunk_depth"] = params["trunk_depth"]
        if params["trunk_hidden_dim"] is not None:
            chemeleon_config["trunk_hidden_dim"] = params["trunk_hidden_dim"]

    config = {
        "data": {
            "data_dir": "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data",
            "splits": None,
            "folds": None,
            "test_file": "assets/dataset/set/local_test.csv",
            "blind_file": "assets/dataset/set/blind_test.csv",
            "output_dir": None,
            "smiles_col": "SMILES",
            "target_cols": [
                "LogD",
                "Log KSOL",
                "Log HLM CLint",
                "Log MLM CLint",
                "Log Caco-2 Permeability Papp A>B",
                "Log Caco-2 Permeability Efflux",
                "Log MPPB",
                "Log MBPB",
                "Log MGMB",
            ],
            "target_weights": [1.0] * 9,
        },
        "model": {
            "type": "chemeleon",
            "chemeleon": chemeleon_config,
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
            "experiment_name": "chemeleon_hpo_ensemble_topk",
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


def save_config(config: dict, output_path: Path) -> None:
    """Save config to YAML file with proper formatting."""
    yaml.add_representer(type(None), represent_none)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)


# =============================================================================
# Visualization Functions
# =============================================================================


class HPOVisualizer:
    """Comprehensive HPO visualization generator for analysis and reporting."""

    # Hyperparameter columns to analyze
    NUMERIC_PARAMS = [
        "config/learning_rate",
        "config/lr_warmup_ratio",
        "config/lr_final_ratio",
        "config/dropout",
        "config/weight_decay",
        "config/ffn_num_layers",
        "config/ffn_hidden_dim",
        "config/batch_size",
        "config/n_experts",
        "config/trunk_depth",
        "config/trunk_hidden_dim",
    ]

    CATEGORICAL_PARAMS = [
        "config/ffn_type",
        "config/batch_norm",
        "config/freeze_encoder",
    ]

    # Display names for cleaner plots
    PARAM_DISPLAY_NAMES = {
        "config/learning_rate": "Learning Rate",
        "config/lr_warmup_ratio": "LR Warmup Ratio",
        "config/lr_final_ratio": "LR Final Ratio",
        "config/dropout": "Dropout",
        "config/weight_decay": "Weight Decay",
        "config/ffn_type": "FFN Type",
        "config/ffn_num_layers": "FFN Layers",
        "config/ffn_hidden_dim": "FFN Hidden Dim",
        "config/batch_size": "Batch Size",
        "config/batch_norm": "Batch Norm",
        "config/n_experts": "N Experts (MoE)",
        "config/trunk_depth": "Trunk Depth (Branched)",
        "config/trunk_hidden_dim": "Trunk Hidden Dim (Branched)",
        "config/freeze_encoder": "Freeze Encoder",
        "val_loss": "Validation Loss",
    }

    def __init__(self, all_trials_df: pd.DataFrame, top_k_df: pd.DataFrame, output_dir: Path):
        self.all_trials = all_trials_df.copy()
        self.top_k = top_k_df.copy()
        self.output_dir = output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
                xlabel = f"log₁₀({self._get_display_name(col)})"
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

        fig.suptitle("Hyperparameter Distributions: All Trials vs Top-K", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "01_parameter_distributions")

    def plot_correlation_heatmap(self) -> Path:
        """Plot correlation heatmap between hyperparameters and val_loss."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Select numeric columns including val_loss
        cols = [c for c in self.NUMERIC_PARAMS + ["val_loss"] if c in self.top_k.columns]
        data = self.top_k[cols].dropna(axis=1, how="all")

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
            "config/weight_decay",
            "config/ffn_num_layers",
            "config/ffn_hidden_dim",
            "config/batch_size",
        ]
        key_params = [c for c in key_params if c in self.top_k.columns]

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
        import seaborn as sns  # noqa: F401 - used implicitly by boxplot palette

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Box plot of val_loss by FFN type (all trials)
        ax = axes[0]
        sns.boxplot(data=self.all_trials, x="config/ffn_type", y="val_loss", ax=ax, palette="Set2")
        ax.set_xlabel("FFN Type")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Val Loss Distribution by FFN Type\n(All Trials)")

        # 2. Count of top-k by FFN type
        ax = axes[1]
        ffn_counts = self.top_k["config/ffn_type"].value_counts()
        colors = sns.color_palette("Set2", len(ffn_counts))
        bars = ax.bar(ffn_counts.index, ffn_counts.values, color=colors)
        ax.set_xlabel("FFN Type")
        ax.set_ylabel("Count")
        ax.set_title(f"FFN Type Distribution\n(Top {len(self.top_k)})")
        for bar, count in zip(bars, ffn_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center", fontsize=10)

        # 3. Val loss vs rank by FFN type
        ax = axes[2]
        for ffn_type in self.top_k["config/ffn_type"].unique():
            subset = self.top_k[self.top_k["config/ffn_type"] == ffn_type]
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
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Learning rate distribution
        ax = axes[0, 0]
        lr_all = self.all_trials["config/learning_rate"].dropna()
        lr_top = self.top_k["config/learning_rate"].dropna()
        ax.hist(np.log10(lr_all), bins=30, alpha=0.5, label="All", color="steelblue", density=True)
        ax.hist(np.log10(lr_top), bins=20, alpha=0.7, label=f"Top {len(self.top_k)}", color="coral", density=True)
        ax.set_xlabel("log₁₀(Learning Rate)")
        ax.set_ylabel("Density")
        ax.set_title("Learning Rate Distribution")
        ax.legend()

        # 2. LR vs val_loss scatter
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.top_k["config/learning_rate"],
            self.top_k["val_loss"],
            c=self.top_k["rank"],
            cmap="viridis_r",
            alpha=0.7,
            s=50,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"LR vs Val Loss (Top {len(self.top_k)})")
        plt.colorbar(scatter, ax=ax, label="Rank")

        # 3. Warmup ratio vs final ratio
        ax = axes[1, 0]
        scatter = ax.scatter(
            self.top_k["config/lr_warmup_ratio"],
            self.top_k["config/lr_final_ratio"],
            c=self.top_k["val_loss"],
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

        # 4. Effective LR range (init to final)
        ax = axes[1, 1]
        lr = self.top_k["config/learning_rate"]
        warmup = self.top_k["config/lr_warmup_ratio"]
        final = self.top_k["config/lr_final_ratio"]
        init_lr = lr / warmup
        final_lr = lr * final

        for i in range(min(20, len(self.top_k))):
            ax.plot([0, 1, 2], [init_lr.iloc[i], lr.iloc[i], final_lr.iloc[i]], alpha=0.5, linewidth=1)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Init LR", "Max LR", "Final LR"])
        ax.set_yscale("log")
        ax.set_ylabel("Learning Rate (log)")
        ax.set_title("LR Schedules (Top 20)")

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
            "config/weight_decay",
            "config/ffn_num_layers",
            "config/ffn_hidden_dim",
            "config/batch_size",
        ]
        params = [c for c in params if c in top_subset.columns]

        # Normalize each parameter to 0-1
        data = top_subset[params].copy()
        for col in params:
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
            ("config/weight_decay", True),
            ("config/ffn_hidden_dim", False),
            ("config/ffn_num_layers", False),
            ("config/batch_size", False),
        ]
        params = [(p, log) for p, log in params if p in self.all_trials.columns]

        n_params = len(params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_params > 1 else [axes]

        for idx, (param, use_log) in enumerate(params):
            ax = axes[idx]

            # Plot all trials in gray
            x_all = self.all_trials[param]
            y_all = self.all_trials["val_loss"]
            ax.scatter(x_all, y_all, alpha=0.1, s=10, color="gray", label="All trials")

            # Highlight top-k
            x_top = self.top_k[param]
            y_top = self.top_k["val_loss"]
            top_label = f"Top {len(self.top_k)}"
            ax.scatter(x_top, y_top, c=self.top_k["rank"], cmap="plasma_r", alpha=0.8, s=30, label=top_label)

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
        data = self.top_k[params].fillna(0)

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
        for i in range(5):
            ax.annotate(f"#{i+1}", (pca_result[i, 0], pca_result[i, 1]), fontsize=8, fontweight="bold")

        # 2. PCA colored by FFN type
        ax = axes[1]
        ffn_types = self.top_k["config/ffn_type"].unique()
        cmap_set1 = plt.get_cmap("Set1")
        colors = cmap_set1(np.linspace(0, 1, len(ffn_types)))
        for ffn_type, color in zip(ffn_types, colors):
            mask = self.top_k["config/ffn_type"] == ffn_type
            ax.scatter(
                pca_result[mask, 0],
                pca_result[mask, 1],
                c=[color],
                label=ffn_type,
                s=50,
                alpha=0.7,
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax.set_title("PCA: Colored by FFN Type")
        ax.legend()

        title = f"Configuration Similarity Analysis (Top {len(self.top_k)})"
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "08_config_similarity_pca")

    def plot_search_space_coverage(self) -> Path:
        """Visualize search space coverage."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 2D histogram: learning_rate vs dropout
        ax = axes[0, 0]
        x = np.log10(self.all_trials["config/learning_rate"].dropna())
        y = self.all_trials["config/dropout"].dropna()
        ax.hist2d(x, y, bins=30, cmap="Blues")
        ax.set_xlabel("log₁₀(Learning Rate)")
        ax.set_ylabel("Dropout")
        ax.set_title("Search Space: LR vs Dropout")

        # 2. 2D histogram: ffn_hidden_dim vs ffn_num_layers
        ax = axes[0, 1]
        x = self.all_trials["config/ffn_hidden_dim"].dropna()
        y = self.all_trials["config/ffn_num_layers"].dropna()
        ax.hist2d(x, y, bins=[20, 5], cmap="Blues")
        ax.set_xlabel("FFN Hidden Dim")
        ax.set_ylabel("FFN Num Layers")
        ax.set_title("Search Space: FFN Architecture")

        # 3. Trial count over time (by source file)
        ax = axes[1, 0]
        source_counts = self.all_trials["source_file"].value_counts().sort_index()
        ax.bar(range(len(source_counts)), source_counts.values, color="steelblue")
        ax.set_xticks(range(len(source_counts)))
        labels = [s.replace("hpo_results_", "").replace(".csv", "") for s in source_counts.index]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("HPO Run")
        ax.set_ylabel("Number of Trials")
        ax.set_title("Trials per HPO Run")

        # 4. Val loss distribution (all vs early-stopped)
        ax = axes[1, 1]
        if "early_stopped" in self.all_trials.columns:
            early_mask = self.all_trials["early_stopped"].fillna(False).astype(bool)
            early_stopped = self.all_trials[early_mask]["val_loss"].dropna()
            completed = self.all_trials[~early_mask]["val_loss"].dropna()
            es_label = f"Early Stopped ({len(early_stopped)})"
            comp_label = f"Completed ({len(completed)})"
            ax.hist(early_stopped, bins=30, alpha=0.5, label=es_label, color="coral", density=True)
            ax.hist(completed, bins=30, alpha=0.5, label=comp_label, color="steelblue", density=True)
            ax.legend()
        else:
            ax.hist(self.all_trials["val_loss"].dropna(), bins=50, color="steelblue", alpha=0.7)
        ax.set_xlabel("Validation Loss")
        ax.set_ylabel("Density")
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
            data = self.top_k[param].dropna()
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
        for idx, row in top_10.iterrows():
            table_data.append(
                [
                    int(row["rank"]),
                    f"{row['val_loss']:.6f}",
                    row["config/ffn_type"],
                    int(row["config/ffn_num_layers"]),
                    int(row["config/ffn_hidden_dim"]),
                    int(row["config/batch_size"]),
                    f"{row['config/learning_rate']:.2e}",
                    f"{row['config/dropout']:.3f}",
                ]
            )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CheMeleon ensemble configs from HPO results")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("chemeleon-hpo"),
        help="Directory containing hpo_results_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/3-hpo-ensemble-chemeleon"),
        help="Output directory for generated configs",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top configs to generate",
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

    # Load and merge HPO results
    merged_df = load_and_merge_hpo_results(args.csv_dir)

    # Sort by val_loss and take top-k
    top_k_df = merged_df.nsmallest(args.top_k, "val_loss").reset_index(drop=True)

    print(f"\nTop {len(top_k_df)} configurations by val_loss:")
    print("-" * 80)
    for i, (_, row) in enumerate(top_k_df.head(10).iterrows()):
        rank = i + 1
        print(
            f"  Rank {rank:3d}: val_loss={row['val_loss']:.6f}, "
            f"ffn_type={row['config/ffn_type']}, "
            f"batch_size={int(row['config/batch_size'])}, "
            f"trial_id={row['trial_id']}"
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
        rank = i + 1  # 1-indexed rank
        params = extract_hyperparams(row)
        config = build_ensemble_config(params, rank)

        output_file = args.output_dir / f"ensemble_chemeleon_hpo_{rank:03d}.yaml"
        save_config(config, output_file)

        if rank <= 5 or rank == len(top_k_df):
            print(f"  Created: {output_file.name} (val_loss={params['val_loss']:.6f})")
        elif rank == 6:
            print("  ...")

    print(f"\nDone! Generated {len(top_k_df)} ensemble config files.")
    print(f"Output directory: {args.output_dir.resolve()}")

    # Generate visualizations
    if not args.skip_plots:
        print("\n" + "=" * 80)
        visualizer = HPOVisualizer(merged_df, top_k_df, args.output_dir)
        visualizer.generate_all_plots()
        print("=" * 80)


if __name__ == "__main__":
    main()
