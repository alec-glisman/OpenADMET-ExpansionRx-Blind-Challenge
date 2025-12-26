"""CLI wrappers for model-related commands (training, ensemble, HPO).

These are thin wrappers that delegate to the existing module-level
`main()` functions so users can call `admet model train ...` instead of
`python -m admet.model.chemprop.model ...`.

Supports multiple model types via the ModelRegistry:
- chemprop: Graph neural network (MPNN)
- chemeleon: Foundation model with pre-trained encoder
- xgboost: XGBoost gradient boosting
- lightgbm: LightGBM gradient boosting
- catboost: CatBoost gradient boosting
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_app = typer.Typer(
    name="model",
    help="Model training and HPO commands",
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)
logger = logging.getLogger("admet.cli.model")


def _register_all_models() -> None:
    """Ensure all model types are registered."""
    try:
        import admet.model.chemprop.adapter  # noqa: F401
    except ImportError:
        pass

    try:
        import admet.model.chemeleon.model  # noqa: F401
    except ImportError:
        pass

    try:
        import admet.model.classical.catboost_model  # noqa: F401
        import admet.model.classical.lightgbm_model  # noqa: F401
        import admet.model.classical.xgboost_model  # noqa: F401
    except ImportError:
        pass


def _run_module_main(module_name: str, args: List[str]) -> None:
    """Run module's main() with given argv (temporarily replaces sys.argv)."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [module_name] + args
        module = importlib.import_module(module_name)
        if not hasattr(module, "main"):
            raise RuntimeError(f"Module {module_name} has no main() function")
        module.main()
    finally:
        sys.argv = old_argv


def load_data_from_config(
    config,
) -> tuple[list[str], np.ndarray, list[str] | None, np.ndarray | None, list[str]]:
    """Load training and optional validation data from config.

    Supports two config formats:
    1. data_dir with train.csv/validation.csv files
    2. data.path with single file (will split if needed)

    Returns
    -------
    tuple
        (train_smiles, train_targets, val_smiles, val_targets, target_cols)
    """
    data_cfg = config.get("data", {})
    smiles_col = data_cfg.get("smiles_col", "SMILES")
    target_cols = list(data_cfg.get("target_cols", []))

    if not target_cols:
        raise ValueError("data.target_cols must be specified in config")

    # Try data_dir first (split structure)
    data_dir = data_cfg.get("data_dir")
    if data_dir:
        data_dir = Path(data_dir)
        train_path = data_dir / "train.csv"
        val_path = data_dir / "validation.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")

        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        train_smiles = train_df[smiles_col].tolist()
        train_targets = train_df[target_cols].values

        val_smiles, val_targets = None, None
        if val_path.exists():
            logger.info(f"Loading validation data from {val_path}")
            val_df = pd.read_csv(val_path)
            val_smiles = val_df[smiles_col].tolist()
            val_targets = val_df[target_cols].values
    else:
        # Fallback to single file
        data_path = data_cfg.get("path")
        if not data_path:
            raise ValueError("Either data.data_dir or data.path must be specified")

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        train_smiles = df[smiles_col].tolist()
        train_targets = df[target_cols].values
        val_smiles, val_targets = None, None

    logger.info(f"Loaded {len(train_smiles)} training samples, {len(target_cols)} targets")
    if val_smiles:
        logger.info(f"Loaded {len(val_smiles)} validation samples")

    return train_smiles, train_targets, val_smiles, val_targets, target_cols


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
) -> dict[str, float]:
    """Compute evaluation metrics for predictions.

    Returns per-target MAE, RMSE, R² and overall metrics.
    """
    metrics = {}

    for i, col in enumerate(target_cols):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() == 0:
            continue

        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        metrics[f"{col}/mae"] = mean_absolute_error(yt, yp)
        metrics[f"{col}/rmse"] = np.sqrt(mean_squared_error(yt, yp))
        metrics[f"{col}/r2"] = r2_score(yt, yp)

    # Overall metrics
    mask = ~np.isnan(y_true)
    metrics["overall/mae"] = mean_absolute_error(y_true[mask], y_pred[mask])
    metrics["overall/rmse"] = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

    return metrics


def _train_model_from_config(cfg, config_path: str, output_path: str | None = None) -> None:
    """Train a model using the ModelRegistry.

    Parameters
    ----------
    cfg : DictConfig
        Model configuration.
    config_path : str
        Path to config file (for logging).
    output_path : str | None
        Override output path for saving model.
    """
    from admet.model.registry import ModelRegistry

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model_type = cfg.get("model", {}).get("type", "unknown")
    logger.info(f"Training {model_type} model from config: {config_path}")

    # Load data
    train_smiles, train_targets, val_smiles, val_targets, target_cols = load_data_from_config(cfg)

    # Create and train model
    model = ModelRegistry.create(cfg)
    logger.info(f"Created {model.__class__.__name__} model")

    model.fit(train_smiles, train_targets, val_smiles=val_smiles, val_y=val_targets)
    logger.info("Training complete")

    # Evaluate on validation data if available
    if val_smiles and val_targets is not None:
        val_preds = model.predict(val_smiles)
        val_metrics = evaluate_predictions(val_targets, val_preds, target_cols)
        logger.info(f"Validation MAE: {val_metrics['overall/mae']:.4f}")
        logger.info(f"Validation RMSE: {val_metrics['overall/rmse']:.4f}")

        for col in target_cols:
            if f"{col}/mae" in val_metrics:
                logger.info(f"  {col}: MAE={val_metrics[f'{col}/mae']:.4f}, " f"R²={val_metrics[f'{col}/r2']:.4f}")

    # Save model
    save_path = output_path or cfg.get("output", {}).get("model_path")
    if save_path and hasattr(model, "save"):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")


@model_app.command("train")
def train(
    config: str = typer.Option(..., "--config", "-c", help="YAML config path"),
    model_type: Optional[str] = typer.Option(
        None,
        "--model-type",
        "-m",
        help="Model type (overrides config). Options: chemprop, chemeleon, xgboost, lightgbm, catboost",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for saved model (overrides config)",
    ),
) -> None:
    """Train a single model using a config YAML.

    The model type is determined from the config's model.type field,
    or can be overridden with --model-type.

    Example:
        admet model train --config configs/0-experiment/chemprop.yaml
        admet model train --config configs/xgboost.yaml --model-type xgboost
    """
    from omegaconf import OmegaConf

    # Ensure all models are registered
    _register_all_models()

    cfg = OmegaConf.load(config)

    # Override model type if specified
    if model_type:
        if "model" not in cfg:
            cfg.model = {}
        cfg.model.type = model_type

    # Get model type from config
    model_cfg = OmegaConf.select(cfg, "model", default={})
    detected_type = OmegaConf.select(model_cfg, "type", default="chemprop") if model_cfg else "chemprop"

    # For chemprop, use existing module main for backward compatibility
    if detected_type == "chemprop":
        _run_module_main("admet.model.chemprop.model", ["--config", config])
    else:
        # Use registry-based training for other model types
        _train_model_from_config(cfg, config, output_path=output)


@model_app.command("train-chemprop")
def train_chemprop(
    config: str = typer.Option(..., "--config", "-c", help="YAML config path"),
) -> None:
    """Train a Chemprop model (legacy command).

    Example:
        admet model train-chemprop --config configs/0-experiment/chemprop.yaml
    """
    _run_module_main("admet.model.chemprop.model", ["--config", config])


@model_app.command("ensemble")
def ensemble(
    config: str = typer.Option(..., "--config", "-c", help="Ensemble config YAML"),
    max_parallel: Optional[int] = typer.Option(None, "--max-parallel", help="Max parallel models"),
) -> None:
    """Train an ensemble using a Chemprop ensemble config.

    Example:
        admet model ensemble --config configs/0-experiment/ensemble_chemprop_production.yaml --max-parallel 2
    """
    args = ["--config", config]
    if max_parallel is not None:
        args += ["--max-parallel", str(max_parallel)]
    _run_module_main("admet.model.chemprop.ensemble", args)


@model_app.command("hpo")
def hpo(
    config: str = typer.Option(..., "--config", "-c", help="HPO config YAML"),
    num_samples: Optional[int] = typer.Option(None, "--num-samples", help="Number of HPO trials"),
) -> None:
    """Run hyperparameter optimization (HPO) using a Chemprop HPO config.

    Example:
        admet model hpo --config configs/1-hpo-single/hpo_chemprop.yaml --num-samples 50
    """
    args = ["--config", config]
    if num_samples is not None:
        args += ["--num-samples", str(num_samples)]
    _run_module_main("admet.model.chemprop.hpo", args)


@model_app.command("list")
def list_models() -> None:
    """List all available model types.

    Example:
        admet model list
    """
    _register_all_models()

    from admet.model.registry import ModelRegistry

    models = ModelRegistry.list_models()

    typer.echo("Available model types:")
    for name in sorted(models):
        model_cls = ModelRegistry.get(name)
        doc = model_cls.__doc__ or "No description"
        first_line = doc.split("\n")[0].strip()
        typer.echo(f"  {name}: {first_line}")
