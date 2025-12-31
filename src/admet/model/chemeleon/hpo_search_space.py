"""Search space builder for CheMeleon Ray Tune hyperparameter optimization.

This module provides functions to convert CheMeleon HPO configuration
dataclasses into Ray Tune search space dictionaries.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from ray import tune

from admet.model.chemeleon.hpo_config import ChemeleonSearchSpaceConfig, ParameterSpace

if TYPE_CHECKING:
    pass


def _build_parameter_space(param: ParameterSpace) -> Any:
    """Convert a ParameterSpace config to a Ray Tune search space object.

    Parameters
    ----------
    param : ParameterSpace
        ParameterSpace configuration specifying the distribution type and bounds.

    Returns
    -------
    Any
        Ray Tune search space object (e.g., tune.uniform, tune.choice, etc.)

    Raises
    ------
    ValueError
        If parameter type is unknown or required fields are missing.
    """
    if param.type == "uniform":
        if param.low is None or param.high is None:
            raise ValueError("uniform distribution requires 'low' and 'high'")
        return tune.uniform(param.low, param.high)

    elif param.type == "loguniform":
        if param.low is None or param.high is None:
            raise ValueError("loguniform distribution requires 'low' and 'high'")
        return tune.loguniform(param.low, param.high)

    elif param.type == "quniform":
        if param.low is None or param.high is None or param.q is None:
            raise ValueError("quniform distribution requires 'low', 'high', and 'q'")
        return tune.quniform(param.low, param.high, param.q)

    elif param.type == "choice":
        if param.values is None:
            raise ValueError("choice distribution requires 'values'")
        return tune.choice(param.values)

    elif param.type == "randint":
        if param.low is None or param.high is None:
            raise ValueError("randint distribution requires 'low' and 'high'")
        return tune.randint(int(param.low), int(param.high))

    elif param.type == "qrandint":
        if param.low is None or param.high is None or param.q is None:
            raise ValueError("qrandint distribution requires 'low', 'high', and 'q'")
        return tune.qrandint(int(param.low), int(param.high), int(param.q))

    else:
        raise ValueError(f"Unknown parameter type: {param.type}")


def build_chemeleon_search_space(
    config: ChemeleonSearchSpaceConfig,
) -> dict[str, Any]:
    """Build a Ray Tune search space dictionary from ChemeleonSearchSpaceConfig.

    This function converts the structured search space config into a flat
    dictionary suitable for Ray Tune's search algorithms. It handles
    conditional parameters (MoE-specific, Branched-specific) using
    tune.sample_from for dynamic sampling.

    Parameters
    ----------
    config : ChemeleonSearchSpaceConfig
        Search space configuration for CheMeleon HPO.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping parameter names to Ray Tune search space objects.

    Examples
    --------
    >>> from admet.model.chemeleon.hpo_config import (
    ...     ChemeleonSearchSpaceConfig, ParameterSpace
    ... )
    >>> config = ChemeleonSearchSpaceConfig(
    ...     learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-3),
    ...     ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts"]),
    ... )
    >>> space = build_chemeleon_search_space(config)
    >>> "learning_rate" in space
    True
    """
    space: dict[str, Any] = {}

    # Simple parameters (no conditions)
    simple_params = [
        "learning_rate",
        "lr_warmup_ratio",
        "lr_final_ratio",
        "warmup_epochs",
        "patience",
        "dropout",
        "weight_decay",
        "ffn_type",
        "ffn_num_layers",
        "ffn_hidden_dim",
        "batch_size",
        "batch_norm",
        "freeze_encoder",
        "unfreeze_encoder_lr_multiplier",
    ]

    for param_name in simple_params:
        param = getattr(config, param_name, None)
        if param is not None:
            space[param_name] = _build_parameter_space(param)

    # Conditional parameters for MoE FFN (n_experts)
    if config.n_experts is not None:
        if config.n_experts.conditional_on == "ffn_type":
            moe_conditional_values = config.n_experts.conditional_values or ["mixture_of_experts"]
            n_experts_config = config.n_experts

            def sample_n_experts(config_dict: dict[str, Any]) -> int | None:
                """Sample n_experts only for MoE FFN types."""
                if config_dict.get("ffn_type") in moe_conditional_values:
                    if n_experts_config.low is not None and n_experts_config.high is not None:
                        return random.randint(int(n_experts_config.low), int(n_experts_config.high))
                return None

            space["n_experts"] = tune.sample_from(sample_n_experts)
        else:
            space["n_experts"] = _build_parameter_space(config.n_experts)

    # Conditional parameters for Branched FFN (trunk_depth)
    if config.trunk_depth is not None:
        if config.trunk_depth.conditional_on == "ffn_type":
            branched_conditional_values = config.trunk_depth.conditional_values or ["branched"]
            trunk_depth_config = config.trunk_depth

            def sample_trunk_depth(config_dict: dict[str, Any]) -> int | None:
                """Sample trunk_depth only for branched FFN types."""
                if config_dict.get("ffn_type") in branched_conditional_values:
                    if trunk_depth_config.low is not None and trunk_depth_config.high is not None:
                        return random.randint(
                            int(trunk_depth_config.low),
                            int(trunk_depth_config.high),
                        )
                return None

            space["trunk_depth"] = tune.sample_from(sample_trunk_depth)
        else:
            space["trunk_depth"] = _build_parameter_space(config.trunk_depth)

    # Conditional parameters for Branched FFN (trunk_hidden_dim)
    if config.trunk_hidden_dim is not None:
        if config.trunk_hidden_dim.conditional_on == "ffn_type":
            branched_conditional_values = config.trunk_hidden_dim.conditional_values or ["branched"]
            trunk_hidden_dim_config = config.trunk_hidden_dim

            def sample_trunk_hidden_dim(config_dict: dict[str, Any]) -> int | None:
                """Sample trunk_hidden_dim only for branched FFN types."""
                if config_dict.get("ffn_type") in branched_conditional_values:
                    if trunk_hidden_dim_config.values is not None:
                        return random.choice(trunk_hidden_dim_config.values)
                    elif trunk_hidden_dim_config.low is not None and trunk_hidden_dim_config.high is not None:
                        return random.randint(
                            int(trunk_hidden_dim_config.low),
                            int(trunk_hidden_dim_config.high),
                        )
                return None

            space["trunk_hidden_dim"] = tune.sample_from(sample_trunk_hidden_dim)
        else:
            space["trunk_hidden_dim"] = _build_parameter_space(config.trunk_hidden_dim)

    # Conditional parameter for encoder unfreezing (only when freeze_encoder=True)
    if config.unfreeze_encoder_epoch is not None:
        if config.unfreeze_encoder_epoch.conditional_on == "freeze_encoder":
            unfreeze_epoch_config = config.unfreeze_encoder_epoch
            conditional_values = config.unfreeze_encoder_epoch.conditional_values or [True]

            def sample_unfreeze_epoch(config_dict: dict[str, Any]) -> int | None:
                """Sample unfreeze_encoder_epoch only when freeze_encoder=True."""
                if config_dict.get("freeze_encoder") in conditional_values:
                    if unfreeze_epoch_config.low is not None and unfreeze_epoch_config.high is not None:
                        return random.randint(
                            int(unfreeze_epoch_config.low),
                            int(unfreeze_epoch_config.high),
                        )
                return None

            space["unfreeze_encoder_epoch"] = tune.sample_from(sample_unfreeze_epoch)
        else:
            space["unfreeze_encoder_epoch"] = _build_parameter_space(config.unfreeze_encoder_epoch)

    return space
