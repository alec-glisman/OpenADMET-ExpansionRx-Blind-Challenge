"""Search space builder for model-agnostic HPO.

Converts HPO search space configurations to Ray Tune format.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from ray import tune

from admet.model.hpo import (
    CatBoostSearchSpace,
    ChemeleonSearchSpace,
    FingerprintSearchSpace,
    HPOSearchSpaceConfig,
    LightGBMSearchSpace,
    ParameterSpace,
    XGBoostSearchSpace,
)


def _build_parameter_space(param: ParameterSpace) -> Any:
    """Convert a ParameterSpace config to a Ray Tune search space object.

    Parameters
    ----------
    param : ParameterSpace
        Parameter space configuration.

    Returns
    -------
    Any
        Ray Tune search space object.

    Raises
    ------
    ValueError
        If parameter type is unknown or required fields are missing.
    """
    if param.type == "uniform":
        if param.low is None or param.high is None:
            msg = "uniform distribution requires 'low' and 'high'"
            raise ValueError(msg)
        return tune.uniform(param.low, param.high)

    elif param.type == "loguniform":
        if param.low is None or param.high is None:
            msg = "loguniform distribution requires 'low' and 'high'"
            raise ValueError(msg)
        return tune.loguniform(param.low, param.high)

    elif param.type == "quniform":
        if param.low is None or param.high is None or param.q is None:
            msg = "quniform distribution requires 'low', 'high', and 'q'"
            raise ValueError(msg)
        return tune.quniform(param.low, param.high, param.q)

    elif param.type == "choice":
        if param.values is None:
            msg = "choice distribution requires 'values'"
            raise ValueError(msg)
        return tune.choice(param.values)

    elif param.type == "randint":
        if param.low is None or param.high is None:
            msg = "randint distribution requires 'low' and 'high'"
            raise ValueError(msg)
        return tune.randint(int(param.low), int(param.high))

    elif param.type == "qrandint":
        if param.low is None or param.high is None or param.q is None:
            msg = "qrandint distribution requires 'low', 'high', and 'q'"
            raise ValueError(msg)
        return tune.qrandint(int(param.low), int(param.high), int(param.q))

    else:
        msg = f"Unknown parameter type: {param.type}"
        raise ValueError(msg)


def _build_dataclass_search_space(config: Any, prefix: str = "") -> dict[str, Any]:
    """Build search space from a dataclass containing ParameterSpace fields.

    Parameters
    ----------
    config : Any
        Dataclass with ParameterSpace fields.
    prefix : str
        Prefix to add to parameter names.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping parameter names to Ray Tune search space objects.
    """
    space: dict[str, Any] = {}

    for field_info in fields(config):
        value = getattr(config, field_info.name)
        if value is None:
            continue

        param_name = f"{prefix}{field_info.name}" if prefix else field_info.name

        if isinstance(value, ParameterSpace):
            space[param_name] = _build_parameter_space(value)
        elif hasattr(value, "__dataclass_fields__"):
            nested_space = _build_dataclass_search_space(value, prefix=f"{param_name}.")
            space.update(nested_space)

    return space


def build_search_space(config: HPOSearchSpaceConfig) -> dict[str, Any]:
    """Build Ray Tune search space from HPO configuration.

    Parameters
    ----------
    config : HPOSearchSpaceConfig
        HPO search space configuration.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping parameter names to Ray Tune search space objects.
    """
    space: dict[str, Any] = {}

    model_type = config.model_type

    # Add fingerprint search space (for classical models)
    if config.fingerprint is not None:
        fp_space = _build_dataclass_search_space(config.fingerprint, prefix="fp_")
        space.update(fp_space)

    # Add model-specific search space
    model_config = None
    if model_type == "xgboost" and config.xgboost is not None:
        model_config = config.xgboost
    elif model_type == "lightgbm" and config.lightgbm is not None:
        model_config = config.lightgbm
    elif model_type == "catboost" and config.catboost is not None:
        model_config = config.catboost
    elif model_type == "chemeleon" and config.chemeleon is not None:
        model_config = config.chemeleon

    if model_config is not None:
        model_space = _build_dataclass_search_space(model_config)
        space.update(model_space)

    return space


def build_xgboost_search_space(config: XGBoostSearchSpace) -> dict[str, Any]:
    """Build search space for XGBoost.

    Parameters
    ----------
    config : XGBoostSearchSpace
        XGBoost search space configuration.

    Returns
    -------
    dict[str, Any]
        Ray Tune search space dictionary.
    """
    return _build_dataclass_search_space(config)


def build_lightgbm_search_space(config: LightGBMSearchSpace) -> dict[str, Any]:
    """Build search space for LightGBM.

    Parameters
    ----------
    config : LightGBMSearchSpace
        LightGBM search space configuration.

    Returns
    -------
    dict[str, Any]
        Ray Tune search space dictionary.
    """
    return _build_dataclass_search_space(config)


def build_catboost_search_space(config: CatBoostSearchSpace) -> dict[str, Any]:
    """Build search space for CatBoost.

    Parameters
    ----------
    config : CatBoostSearchSpace
        CatBoost search space configuration.

    Returns
    -------
    dict[str, Any]
        Ray Tune search space dictionary.
    """
    return _build_dataclass_search_space(config)


def build_chemeleon_search_space(config: ChemeleonSearchSpace) -> dict[str, Any]:
    """Build search space for Chemeleon.

    Parameters
    ----------
    config : ChemeleonSearchSpace
        Chemeleon search space configuration.

    Returns
    -------
    dict[str, Any]
        Ray Tune search space dictionary.
    """
    return _build_dataclass_search_space(config)


def build_fingerprint_search_space(
    config: FingerprintSearchSpace,
) -> dict[str, Any]:
    """Build search space for fingerprint configuration.

    Parameters
    ----------
    config : FingerprintSearchSpace
        Fingerprint search space configuration.

    Returns
    -------
    dict[str, Any]
        Ray Tune search space dictionary.
    """
    return _build_dataclass_search_space(config, prefix="fp_")
