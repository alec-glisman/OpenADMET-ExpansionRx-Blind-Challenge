"""Model-agnostic HPO search space definitions.

This module provides search space configurations for hyperparameter optimization
across all supported model types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class ParameterSpace:
    """Configuration for a single hyperparameter search space.

    Attributes:
        type: Distribution type - one of "uniform", "loguniform", "choice",
              "quniform", "randint", "qrandint"
        low: Lower bound for uniform/loguniform/quniform/randint/qrandint
        high: Upper bound for uniform/loguniform/quniform/randint/qrandint
        values: List of values for choice distribution
        q: Quantization step for quniform/qrandint distribution
    """

    type: str = MISSING
    low: float | None = None
    high: float | None = None
    values: list[Any] | None = None
    q: float | None = None


@dataclass
class XGBoostSearchSpace:
    """Search space for XGBoost hyperparameters.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        min_child_weight: Minimum sum of instance weight in a child.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns by tree.
        gamma: Minimum loss reduction for split.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
    """

    n_estimators: ParameterSpace | None = None
    max_depth: ParameterSpace | None = None
    learning_rate: ParameterSpace | None = None
    min_child_weight: ParameterSpace | None = None
    subsample: ParameterSpace | None = None
    colsample_bytree: ParameterSpace | None = None
    gamma: ParameterSpace | None = None
    reg_alpha: ParameterSpace | None = None
    reg_lambda: ParameterSpace | None = None


@dataclass
class LightGBMSearchSpace:
    """Search space for LightGBM hyperparameters.

    Attributes:
        n_estimators: Number of boosting rounds.
        num_leaves: Maximum number of leaves in one tree.
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth (-1 for no limit).
        min_child_samples: Minimum samples in a leaf.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns by tree.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
    """

    n_estimators: ParameterSpace | None = None
    num_leaves: ParameterSpace | None = None
    learning_rate: ParameterSpace | None = None
    max_depth: ParameterSpace | None = None
    min_child_samples: ParameterSpace | None = None
    subsample: ParameterSpace | None = None
    colsample_bytree: ParameterSpace | None = None
    reg_alpha: ParameterSpace | None = None
    reg_lambda: ParameterSpace | None = None


@dataclass
class CatBoostSearchSpace:
    """Search space for CatBoost hyperparameters.

    Attributes:
        iterations: Number of boosting rounds.
        depth: Tree depth.
        learning_rate: Boosting learning rate.
        l2_leaf_reg: L2 regularization coefficient.
        bagging_temperature: Bayesian bootstrap parameter.
        random_strength: Random feature split score coefficient.
        border_count: Number of splits for numerical features.
    """

    iterations: ParameterSpace | None = None
    depth: ParameterSpace | None = None
    learning_rate: ParameterSpace | None = None
    l2_leaf_reg: ParameterSpace | None = None
    bagging_temperature: ParameterSpace | None = None
    random_strength: ParameterSpace | None = None
    border_count: ParameterSpace | None = None


@dataclass
class ChemeleonSearchSpace:
    """Search space for Chemeleon hyperparameters.

    Attributes:
        ffn_hidden_dim: Hidden dimension for FFN layers.
        ffn_num_layers: Number of FFN layers.
        dropout: Dropout probability.
        learning_rate: Learning rate.
        freeze_encoder: Whether to freeze encoder.
        unfreeze_encoder_epoch: Epoch to unfreeze encoder.
        unfreeze_encoder_lr_multiplier: LR multiplier for unfrozen encoder.
    """

    ffn_hidden_dim: ParameterSpace | None = None
    ffn_num_layers: ParameterSpace | None = None
    dropout: ParameterSpace | None = None
    learning_rate: ParameterSpace | None = None
    freeze_encoder: ParameterSpace | None = None
    unfreeze_encoder_epoch: ParameterSpace | None = None
    unfreeze_encoder_lr_multiplier: ParameterSpace | None = None


@dataclass
class FingerprintSearchSpace:
    """Search space for fingerprint configuration.

    Attributes:
        fp_type: Fingerprint type (morgan, rdkit, maccs, mordred).
        morgan_radius: Morgan fingerprint radius.
        morgan_n_bits: Morgan fingerprint bit size.
        rdkit_max_path: RDKit fingerprint max path length.
        rdkit_n_bits: RDKit fingerprint bit size.
    """

    fp_type: ParameterSpace | None = None
    morgan_radius: ParameterSpace | None = None
    morgan_n_bits: ParameterSpace | None = None
    rdkit_max_path: ParameterSpace | None = None
    rdkit_n_bits: ParameterSpace | None = None


@dataclass
class HPOSearchSpaceConfig:
    """Unified search space configuration for all model types.

    Select the appropriate model-specific search space based on model.type.

    Attributes:
        model_type: Type of model (determines which search space to use).
        fingerprint: Fingerprint configuration search space (for classical models).
        xgboost: XGBoost-specific search space.
        lightgbm: LightGBM-specific search space.
        catboost: CatBoost-specific search space.
        chemeleon: Chemeleon-specific search space.
    """

    model_type: str = "xgboost"
    fingerprint: FingerprintSearchSpace | None = None
    xgboost: XGBoostSearchSpace | None = None
    lightgbm: LightGBMSearchSpace | None = None
    catboost: CatBoostSearchSpace | None = None
    chemeleon: ChemeleonSearchSpace | None = None


def get_default_xgboost_search_space() -> XGBoostSearchSpace:
    """Get default XGBoost search space."""
    return XGBoostSearchSpace(
        n_estimators=ParameterSpace(type="randint", low=50, high=500),
        max_depth=ParameterSpace(type="randint", low=3, high=10),
        learning_rate=ParameterSpace(type="loguniform", low=0.01, high=0.3),
        min_child_weight=ParameterSpace(type="randint", low=1, high=10),
        subsample=ParameterSpace(type="uniform", low=0.5, high=1.0),
        colsample_bytree=ParameterSpace(type="uniform", low=0.5, high=1.0),
        gamma=ParameterSpace(type="loguniform", low=1e-8, high=1.0),
        reg_alpha=ParameterSpace(type="loguniform", low=1e-8, high=10.0),
        reg_lambda=ParameterSpace(type="loguniform", low=1e-8, high=10.0),
    )


def get_default_lightgbm_search_space() -> LightGBMSearchSpace:
    """Get default LightGBM search space."""
    return LightGBMSearchSpace(
        n_estimators=ParameterSpace(type="randint", low=50, high=500),
        num_leaves=ParameterSpace(type="randint", low=15, high=127),
        learning_rate=ParameterSpace(type="loguniform", low=0.01, high=0.3),
        max_depth=ParameterSpace(type="randint", low=-1, high=15),
        min_child_samples=ParameterSpace(type="randint", low=5, high=100),
        subsample=ParameterSpace(type="uniform", low=0.5, high=1.0),
        colsample_bytree=ParameterSpace(type="uniform", low=0.5, high=1.0),
        reg_alpha=ParameterSpace(type="loguniform", low=1e-8, high=10.0),
        reg_lambda=ParameterSpace(type="loguniform", low=1e-8, high=10.0),
    )


def get_default_catboost_search_space() -> CatBoostSearchSpace:
    """Get default CatBoost search space."""
    return CatBoostSearchSpace(
        iterations=ParameterSpace(type="randint", low=50, high=500),
        depth=ParameterSpace(type="randint", low=4, high=10),
        learning_rate=ParameterSpace(type="loguniform", low=0.01, high=0.3),
        l2_leaf_reg=ParameterSpace(type="loguniform", low=1.0, high=10.0),
        bagging_temperature=ParameterSpace(type="uniform", low=0.0, high=1.0),
        random_strength=ParameterSpace(type="uniform", low=0.0, high=1.0),
        border_count=ParameterSpace(type="randint", low=32, high=255),
    )


def get_default_chemeleon_search_space() -> ChemeleonSearchSpace:
    """Get default Chemeleon search space."""
    return ChemeleonSearchSpace(
        ffn_hidden_dim=ParameterSpace(type="choice", values=[128, 256, 512]),
        ffn_num_layers=ParameterSpace(type="randint", low=1, high=4),
        dropout=ParameterSpace(type="uniform", low=0.0, high=0.5),
        learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
        freeze_encoder=ParameterSpace(type="choice", values=[True, False]),
        unfreeze_encoder_epoch=ParameterSpace(type="randint", low=1, high=10),
        unfreeze_encoder_lr_multiplier=ParameterSpace(type="loguniform", low=0.01, high=1.0),
    )


def get_default_fingerprint_search_space() -> FingerprintSearchSpace:
    """Get default fingerprint search space."""
    return FingerprintSearchSpace(
        fp_type=ParameterSpace(type="choice", values=["morgan", "rdkit", "maccs"]),
        morgan_radius=ParameterSpace(type="randint", low=2, high=4),
        morgan_n_bits=ParameterSpace(type="choice", values=[1024, 2048, 4096]),
    )


def get_search_space_for_model(
    model_type: str,
) -> XGBoostSearchSpace | LightGBMSearchSpace | CatBoostSearchSpace | ChemeleonSearchSpace:
    """Get default search space for a model type.

    Parameters
    ----------
    model_type : str
        Model type identifier.

    Returns
    -------
    Search space for the model type.

    Raises
    ------
    ValueError
        If model type is not supported for HPO.
    """
    search_spaces = {
        "xgboost": get_default_xgboost_search_space,
        "lightgbm": get_default_lightgbm_search_space,
        "catboost": get_default_catboost_search_space,
        "chemeleon": get_default_chemeleon_search_space,
    }

    if model_type not in search_spaces:
        msg = f"HPO not supported for model type: {model_type}"
        raise ValueError(msg)

    result = search_spaces[model_type]()
    # Type narrowing: we've validated model_type is in search_spaces
    return result  # type: ignore[return-value]
