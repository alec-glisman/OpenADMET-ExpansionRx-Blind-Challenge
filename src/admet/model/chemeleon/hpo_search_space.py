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


def _build_parameter_space(param: ParameterSpace | dict[str, Any]) -> Any:
    """Convert a ParameterSpace config to a Ray Tune search space object.

    Parameters
    ----------
    param : ParameterSpace | dict[str, Any]
        ParameterSpace configuration specifying the distribution type and bounds.
        Can be a ParameterSpace dataclass instance or a dict with the same fields.

    Returns
    -------
    Any
        Ray Tune search space object (e.g., tune.uniform, tune.choice, etc.)

    Raises
    ------
    ValueError
        If parameter type is unknown or required fields are missing.
    """
    # Handle both ParameterSpace objects and plain dicts
    if isinstance(param, dict):
        param_type = param.get("type")
        param_low = param.get("low")
        param_high = param.get("high")
        param_values = param.get("values")
        param_q = param.get("q")
    else:
        param_type = param.type
        param_low = param.low
        param_high = param.high
        param_values = param.values
        param_q = param.q

    if param_type == "uniform":
        if param_low is None or param_high is None:
            raise ValueError("uniform distribution requires 'low' and 'high'")
        return tune.uniform(param_low, param_high)

    elif param_type == "loguniform":
        if param_low is None or param_high is None:
            raise ValueError("loguniform distribution requires 'low' and 'high'")
        return tune.loguniform(param_low, param_high)

    elif param_type == "quniform":
        if param_low is None or param_high is None or param_q is None:
            raise ValueError("quniform distribution requires 'low', 'high', and 'q'")
        return tune.quniform(param_low, param_high, param_q)

    elif param_type == "choice":
        if param_values is None:
            raise ValueError("choice distribution requires 'values'")
        return tune.choice(param_values)

    elif param_type == "randint":
        if param_low is None or param_high is None:
            raise ValueError("randint distribution requires 'low' and 'high'")
        return tune.randint(int(param_low), int(param_high))

    elif param_type == "qrandint":
        if param_low is None or param_high is None or param_q is None:
            raise ValueError("qrandint distribution requires 'low', 'high', and 'q'")
        return tune.qrandint(int(param_low), int(param_high), int(param_q))

    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def build_chemeleon_search_space(
    config: ChemeleonSearchSpaceConfig,
    target_columns: list[str] | None = None,
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
    target_columns : list[str] | None
        Optional list of target column names. If provided, per-target
        weight parameters will be added to the search space.

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
    # target_columns reserved for future per-target weight search space support
    _ = target_columns

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
    ]

    for param_name in simple_params:
        param = getattr(config, param_name, None)
        if param is not None:
            space[param_name] = _build_parameter_space(param)

    # Conditional parameter: unfreeze_encoder_lr_multiplier (only when freeze_encoder=True)
    if config.unfreeze_encoder_lr_multiplier is not None:
        if config.unfreeze_encoder_lr_multiplier.conditional_on == "freeze_encoder":
            lr_mult_config = config.unfreeze_encoder_lr_multiplier
            conditional_values = config.unfreeze_encoder_lr_multiplier.conditional_values or [True]

            def sample_unfreeze_lr_multiplier(config_dict: dict[str, Any]) -> float | None:
                """Sample unfreeze_encoder_lr_multiplier only when freeze_encoder=True."""
                if config_dict.get("freeze_encoder") in conditional_values:
                    if lr_mult_config.type == "loguniform":
                        import math
                        log_low = math.log(lr_mult_config.low)
                        log_high = math.log(lr_mult_config.high)
                        return math.exp(random.uniform(log_low, log_high))
                    elif lr_mult_config.type == "uniform":
                        return random.uniform(lr_mult_config.low, lr_mult_config.high)
                return None

            space["unfreeze_encoder_lr_multiplier"] = tune.sample_from(sample_unfreeze_lr_multiplier)
        else:
            # Non-conditional
            space["unfreeze_encoder_lr_multiplier"] = _build_parameter_space(config.unfreeze_encoder_lr_multiplier)

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

    # Joint sampling parameters (if specified in search space)
    if config.joint_sampling is not None and isinstance(config.joint_sampling, dict):
        joint_config = config.joint_sampling

        # joint_sampling.enabled
        if "enabled" in joint_config and joint_config["enabled"] is not None:
            space["joint_sampling_enabled"] = _build_parameter_space(joint_config["enabled"])

        # joint_sampling.task_oversampling.alpha (conditional on joint_sampling_enabled)
        if "task_oversampling" in joint_config and joint_config["task_oversampling"] is not None:
            task_oversample_config = joint_config["task_oversampling"]
            if "alpha" in task_oversample_config and task_oversample_config["alpha"] is not None:
                # Make alpha conditional on joint_sampling_enabled
                alpha_param = task_oversample_config["alpha"]

                def sample_joint_sampling_alpha(config_dict: dict[str, Any]) -> float | None:
                    """Sample joint_sampling_alpha only when joint_sampling_enabled is True."""
                    if config_dict.get("joint_sampling_enabled", False):
                        # Sample using the parameter space definition
                        if alpha_param.type == "uniform":
                            return random.uniform(alpha_param.low, alpha_param.high)
                        elif alpha_param.type == "loguniform":
                            import math
                            log_low = math.log(alpha_param.low)
                            log_high = math.log(alpha_param.high)
                            return math.exp(random.uniform(log_low, log_high))
                        elif alpha_param.type == "choice":
                            return random.choice(list(alpha_param.values))
                    return None

                space["joint_sampling_alpha"] = tune.sample_from(sample_joint_sampling_alpha)

    return space
