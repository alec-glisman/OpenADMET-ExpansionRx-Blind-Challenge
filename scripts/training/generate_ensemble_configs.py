"""
Generate 100 ensemble configuration files from top HPO results.
"""

import json
import math
from pathlib import Path
from typing import Optional

ENSEMBLE_DATA_DIR = "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data"


def load_top_configs(filepath: str) -> list[dict]:
    """Load top k configurations from JSON file."""
    with open(filepath, "r") as f:
        configs = json.load(f)
    return configs


def generate_ensemble_config(hpo_config: dict, rank: int) -> str:
    """Generate ensemble YAML configuration from HPO config."""

    # Extract hyperparameters
    learning_rate = hpo_config["learning_rate"]
    lr_warmup_ratio = hpo_config["lr_warmup_ratio"]
    lr_final_ratio = hpo_config["lr_final_ratio"]
    dropout = hpo_config["dropout"]
    depth = hpo_config["depth"]
    message_hidden_dim = hpo_config["message_hidden_dim"]
    ffn_num_layers = hpo_config["ffn_num_layers"]
    ffn_hidden_dim = hpo_config["ffn_hidden_dim"]
    batch_size = hpo_config["batch_size"]
    ffn_type = hpo_config["ffn_type"]

    # Map "mlp" to "regression" as the model doesn't support "mlp"
    if ffn_type == "mlp":
        ffn_type = "regression"
    aggregation = hpo_config["aggregation"]
    task_sampling_alpha = hpo_config["task_sampling_alpha"]

    # Handle NaN or missing values for conditional parameters
    n_experts: Optional[float] = hpo_config.get("n_experts")
    trunk_depth: Optional[float] = hpo_config.get("trunk_depth")
    trunk_hidden_dim: Optional[float] = hpo_config.get("trunk_hidden_dim")

    # Compute safe defaults for integer-valued hyperparameters
    if n_experts is not None and not math.isnan(n_experts):
        n_experts_val = int(n_experts)
    else:
        n_experts_val = 4

    if trunk_depth is not None and not math.isnan(trunk_depth):
        trunk_n_layers_val = int(trunk_depth)
    else:
        trunk_n_layers_val = 2

    if trunk_hidden_dim is not None and not math.isnan(trunk_hidden_dim):
        trunk_hidden_dim_val = int(trunk_hidden_dim)
    else:
        trunk_hidden_dim_val = 500

    # Calculate derived learning rates
    init_lr = learning_rate * lr_warmup_ratio
    max_lr = learning_rate
    final_lr = learning_rate * lr_final_ratio

    # Get metric value
    metric_value = hpo_config.get("_metric_value", "N/A")

    # Build YAML content
    yaml_lines = [
        f"# Ensemble ChempropModel Configuration (HPO Rank {rank})",
        "# =====================================",
        f"# Generated from top_k_configs.json rank {rank}",
        f"# Validation MAE: {metric_value}",
        "#",
        "# This YAML file configures ensemble training across multiple splits and folds.",
        "# The script will automatically discover split_*/fold_*/ subdirectories and",
        "# train a model on each, then aggregate predictions with uncertainty estimates.",
        "#",
        "# Usage:",
        "#   python -m admet.model.chemprop.ensemble --config \\",
        f"#       configs/2-hpo-ensemble/ensemble_chemprop_hpo_{rank:03d}.yaml",
        "#   python -m admet.model.chemprop.ensemble -c \\",
        f"#       configs/2-hpo-ensemble/ensemble_chemprop_hpo_{rank:03d}.yaml --max-parallel 2",
        "",
        "# Data configuration for ensemble",
        "data:",
        f'  data_dir: "{ENSEMBLE_DATA_DIR}"',
        "  splits: null",
        "  folds: null",
        "",
        '  test_file: "assets/dataset/set/local_test.csv"',
        '  blind_file: "assets/dataset/set/blind_test.csv"',
        "  output_dir: null",
        "",
        '  smiles_col: "SMILES"',
        "  target_cols:",
        '    - "LogD"',
        '    - "Log KSOL"',
        '    - "Log HLM CLint"',
        '    - "Log MLM CLint"',
        '    - "Log Caco-2 Permeability Papp A>B"',
        '    - "Log Caco-2 Permeability Efflux"',
        '    - "Log MPPB"',
        '    - "Log MBPB"',
        '    - "Log MGMB"',
        "  target_weights:",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "    - 1.0",
        "",
        "model:",
        f"  depth: {depth}",
        f"  message_hidden_dim: {message_hidden_dim}",
        f'  aggregation: "{aggregation}"',
    ]

    # Add FFN type-specific configuration
    if ffn_type == "mlp":
        yaml_lines.extend(
            [
                '  ffn_type: "mlp"',
                f"  num_layers: {ffn_num_layers}",
                f"  hidden_dim: {ffn_hidden_dim}",
            ]
        )
    elif ffn_type == "moe":
        yaml_lines.extend(
            [
                '  ffn_type: "moe"',
                f"  num_layers: {ffn_num_layers}",
                f"  hidden_dim: {ffn_hidden_dim}",
                f"  n_experts: {n_experts_val}",
            ]
        )
    elif ffn_type == "branched":
        yaml_lines.extend(
            [
                '  ffn_type: "branched"',
                f"  num_layers: {ffn_num_layers}",
                f"  hidden_dim: {ffn_hidden_dim}",
                f"  trunk_n_layers: {trunk_n_layers_val}",
                f"  trunk_hidden_dim: {trunk_hidden_dim_val}",
            ]
        )
    else:  # regression (legacy)
        yaml_lines.extend(
            [
                '  ffn_type: "regression"',
                f"  num_layers: {ffn_num_layers}",
                f"  hidden_dim: {ffn_hidden_dim}",
            ]
        )

    yaml_lines.extend(
        [
            f"  dropout: {dropout}",
            "  batch_norm: true",
            "",
            "optimization:",
            '  criterion: "MAE"',
            f"  init_lr: {init_lr:.2e}",
            f"  max_lr: {max_lr:.2e}",
            f"  final_lr: {final_lr:.2e}",
            "  warmup_epochs: 5",
            "  max_epochs: 150",
            "  patience: 15",
            f"  batch_size: {batch_size}",
            "  num_workers: 0",
            "  seed: 12345",
            "  progress_bar: false",
            "",
            "mlflow:",
            "  tracking: true",
            '  tracking_uri: "http://127.0.0.1:8084"',
            '  experiment_name: "ensemble_chemprop_hpo_topk"',
            f'  run_name: "rank_{rank:03d}"',
            "  nested: true",
            "",
            "max_parallel: 1",
            "ray_num_cpus: null",
            "ray_num_gpus: null",
            "",
            "# Joint Sampling Configuration",
            "# =============================",
            "# Unified two-stage sampling combining task-aware oversampling with curriculum learning.",
            "# Stage 1: Task selection with inverse-power weighting (alpha)",
            "# Stage 2: Within-task sampling weighted by curriculum phase",
            "joint_sampling:",
            "  enabled: true",
            "",
            "  # Task-aware oversampling (Stage 1)",
            "  task_oversampling:",
            f"    alpha: {task_sampling_alpha}  # [0, 1] - task rebalancing strength",
            "",
            "  # Curriculum learning (Stage 2) - disabled by default",
            "  # Enable when training on mixed-quality datasets",
            "  curriculum:",
            "    enabled: false",
            '    quality_col: "Quality"',
            "    qualities:",
            '      - "high"',
            '      - "medium"',
            '      - "low"',
            "    patience: 5",
            "    count_normalize: true",
            "    min_high_quality_proportion: 0.25",
            '    strategy: "sampled"',
            "    reset_early_stopping_on_phase_change: false",
            "    log_per_quality_metrics: true",
            "",
            "  # Global sampling settings",
            "  seed: 42",
            "  increment_seed_per_epoch: true",
            "  log_to_mlflow: true",
            "",
            "# =============================================================================",
            "# Inter-Task Affinity Configuration (Paper-Accurate Implementation)",
            "# =============================================================================",
            "# Computes inter-task affinity during training using the lookahead method from:",
            '# "Efficiently Identifying Task Groupings for Multi-Task Learning"',
            "# (Fifty et al., NeurIPS 2021, https://arxiv.org/abs/2109.04617)",
            "#",
            "# Formula: Z^t_{ij} = 1 - L_j(θ^{t+1}_{s|i}) / L_j(θ^t_s)",
            "# =============================================================================",
            "inter_task_affinity:",
            "  enabled: false",
            "  compute_every_n_steps: 2",
            "  log_every_n_steps: 10",
            "  log_epoch_summary: true",
            "  log_step_matrices: false",
            "  lookahead_lr: 0.001",
            "  use_optimizer_lr: true",
            "  exclude_param_patterns:",
            "    - predictor",
            "    - ffn",
            "    - output",
            "    - head",
            "    - readout",
            "  log_to_mlflow: true",
        ]
    )

    return "\n".join(yaml_lines) + "\n"


def main():
    """Main function to generate all ensemble configs."""
    # Load top configs
    top_configs = load_top_configs("top_k_configs.json")

    # Create configs directory if it doesn't exist
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Generate config files
    for i, config in enumerate(top_configs[:100], start=1):
        yaml_content = generate_ensemble_config(config, i)
        output_path = configs_dir / f"ensemble_chemprop_hpo_{i:03d}.yaml"

        with open(output_path, "w") as f:
            f.write(yaml_content)

        print(f"Generated {output_path}")

    print("\nSuccessfully generated 100 ensemble configuration files")


if __name__ == "__main__":
    main()
