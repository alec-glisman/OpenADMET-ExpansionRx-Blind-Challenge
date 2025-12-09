"""Search space builder for Ray Tune hyperparameter optimization.

This module provides functions to convert HPO configuration dataclasses
into Ray Tune search space dictionaries.
"""

import random
from typing import Any

from ray import tune

from admet.model.chemprop.hpo_config import ParameterSpace, SearchSpaceConfig


def _build_parameter_space(param: ParameterSpace) -> Any:
    """Convert a ParameterSpace config to a Ray Tune search space object.

    Args:
        param: ParameterSpace configuration specifying the distribution type
            and bounds.

    Returns:
        Ray Tune search space object (e.g., tune.uniform, tune.choice, etc.)

    Raises:
        ValueError: If parameter type is unknown or required fields are missing.
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


def build_search_space(
    config: SearchSpaceConfig,
    target_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build a Ray Tune search space dictionary from SearchSpaceConfig.

    This function converts the structured SearchSpaceConfig into a flat
    dictionary suitable for Ray Tune's search algorithms. It handles
    conditional parameters (e.g., MoE-specific parameters) using
    tune.sample_from for dynamic sampling.

    Args:
        config: SearchSpaceConfig containing parameter space definitions.
        target_columns: List of target column names. Required if target_weights
            is specified in config, to create per-target weight parameters.

    Returns:
        Dictionary mapping parameter names to Ray Tune search space objects.

    Example:
        >>> from admet.model.chemprop.hpo_config import SearchSpaceConfig, ParameterSpace
        >>> config = SearchSpaceConfig(
        ...     learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
        ...     batch_size=ParameterSpace(type="choice", values=[32, 64, 128]),
        ... )
        >>> space = build_search_space(config)
        >>> "learning_rate" in space
        True
    """
    space: dict[str, Any] = {}

    # Simple parameters (no conditions)
    # Note: weight_decay is not included as it's not currently applied in ChempropHyperparams
    simple_params = [
        "learning_rate",
        "lr_warmup_ratio",
        "lr_final_ratio",
        "warmup_epochs",
        "patience",
        "dropout",
        "depth",
        "message_hidden_dim",
        "hidden_dim",  # Deprecated: use message_hidden_dim + ffn_hidden_dim
        "ffn_num_layers",
        "ffn_hidden_dim",
        "batch_size",
        "ffn_type",
        "aggregation",
        "aggregation_norm",
        "task_sampling_alpha",
    ]

    for param_name in simple_params:
        param = getattr(config, param_name, None)
        if param is not None:
            space[param_name] = _build_parameter_space(param)

    # Conditional parameters for MoE FFN
    if config.n_experts is not None:
        if config.n_experts.conditional_on == "ffn_type":
            # Sample only when ffn_type is in conditional_values
            conditional_values = config.n_experts.conditional_values or ["moe"]

            def sample_n_experts(config_dict: dict[str, Any]) -> int | None:
                """Sample n_experts only for MoE FFN types."""
                if config_dict.get("ffn_type") in conditional_values:
                    param = config.n_experts
                    if param is not None and param.low is not None and param.high is not None:
                        q = param.q if param.q is not None else 1
                        return int(random.randint(int(param.low), int(param.high)) // q * q)
                return None

            space["n_experts"] = tune.sample_from(sample_n_experts)
        else:
            # Non-conditional n_experts
            space["n_experts"] = _build_parameter_space(config.n_experts)

    # Conditional parameters for Branched FFN
    if config.trunk_depth is not None:
        if config.trunk_depth.conditional_on == "ffn_type":
            conditional_values = config.trunk_depth.conditional_values or ["branched"]

            def sample_trunk_depth(config_dict: dict[str, Any]) -> int | None:
                """Sample trunk_depth only for branched FFN types."""
                if config_dict.get("ffn_type") in conditional_values:
                    param = config.trunk_depth
                    if param is not None and param.low is not None and param.high is not None:
                        q = param.q if param.q is not None else 1
                        return int(random.randint(int(param.low), int(param.high)) // q * q)
                return None

            space["trunk_depth"] = tune.sample_from(sample_trunk_depth)
        else:
            space["trunk_depth"] = _build_parameter_space(config.trunk_depth)

    if config.trunk_hidden_dim is not None:
        if config.trunk_hidden_dim.conditional_on == "ffn_type":
            conditional_values = config.trunk_hidden_dim.conditional_values or ["branched"]

            def sample_trunk_hidden_dim(config_dict: dict[str, Any]) -> int | None:
                """Sample trunk_hidden_dim only for branched FFN types."""
                if config_dict.get("ffn_type") in conditional_values:
                    param = config.trunk_hidden_dim
                    if param is not None and param.values is not None:
                        return random.choice(param.values)
                return None

            space["trunk_hidden_dim"] = tune.sample_from(sample_trunk_hidden_dim)
        else:
            space["trunk_hidden_dim"] = _build_parameter_space(config.trunk_hidden_dim)

    # Target weights - one parameter per target column
    if config.target_weights is not None and target_columns is not None:
        for target in target_columns:
            # Create a safe parameter name from the target column
            safe_name = target.replace(" ", "_").replace(">", "gt").replace("<", "lt")
            param_name = f"target_weight_{safe_name}"
            space[param_name] = _build_parameter_space(config.target_weights)

    return space


def get_default_search_space() -> SearchSpaceConfig:
    """Get a default search space configuration for Chemprop HPO.

    This provides reasonable default ranges for common hyperparameters
    based on empirical observations and Chemprop documentation.

    Note: weight_decay is not included as it requires extending
    ChempropHyperparams to support AdamW weight decay configuration.

    Returns:
        SearchSpaceConfig with default parameter ranges.
    """
    return SearchSpaceConfig(
        # Learning rate schedule
        learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
        lr_warmup_ratio=ParameterSpace(type="uniform", low=0.01, high=0.2),
        lr_final_ratio=ParameterSpace(type="uniform", low=0.01, high=0.2),
        warmup_epochs=ParameterSpace(type="choice", values=[2, 3, 5, 8]),
        patience=ParameterSpace(type="choice", values=[10, 15, 20, 25]),
        # Regularization
        dropout=ParameterSpace(type="uniform", low=0.0, high=0.4),
        # Message passing (MPNN)
        depth=ParameterSpace(type="choice", values=[2, 3, 4, 5, 6]),
        message_hidden_dim=ParameterSpace(type="choice", values=[256, 512, 768, 1024]),
        # FFN architecture
        ffn_num_layers=ParameterSpace(type="choice", values=[1, 2, 3]),
        ffn_hidden_dim=ParameterSpace(type="choice", values=[256, 512, 768, 1024]),
        batch_size=ParameterSpace(type="choice", values=[32, 64, 128, 256]),
        ffn_type=ParameterSpace(type="choice", values=["mlp", "moe", "branched"]),
        n_experts=ParameterSpace(
            type="choice",
            values=[2, 4, 8],
            conditional_on="ffn_type",
            conditional_values=["moe"],
        ),
        # Aggregation
        aggregation=ParameterSpace(type="choice", values=["mean", "sum", "norm"]),
        # Task weighting
        target_weights=ParameterSpace(type="uniform", low=0.05, high=50.0),
    )
