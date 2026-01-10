"""Automatic search range refinement for multi-phase HPO.

This module provides utilities to automatically extract and narrow hyperparameter
search ranges from previous HPO phases, enabling efficient Phase 3 refinement
without manual TODO updates.

Example usage:
    from admet.model.hpo_range_refinement import refine_search_space

    # Load top configs from Phase 2
    refined_config = refine_search_space(
        base_config=phase3_config.search_space,
        previous_phase_dir="/path/to/phase2/output",
        top_k=10,
        margin_factor=0.5,  # 50% margin around observed ranges
    )
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from admet.model.chemprop.hpo_config import ParameterSpace, SearchSpaceConfig

logger = logging.getLogger("admet.model.hpo_range_refinement")


@dataclass
class RefinementConfig:
    """Configuration for automatic search space refinement.

    Attributes:
        enabled: Whether to enable automatic refinement from previous phase
        previous_phase_dir: Directory containing previous phase HPO outputs
            (must contain top_k_configs.json or hpo_results.csv)
        top_k: Number of top configs to use for range estimation
        margin_factor: Factor to expand ranges beyond observed min/max (0.0-1.0)
            - 0.0: Use exact observed min/max
            - 0.5: Add 50% margin on each side (log-scale for loguniform)
            - 1.0: Double the range on each side
        min_samples: Minimum samples required for statistical estimates
        use_percentiles: Use percentile-based ranges (10th-90th) instead of min/max
        percentile_low: Lower percentile when use_percentiles=True (default: 10)
        percentile_high: Upper percentile when use_percentiles=True (default: 90)
        params_to_refine: List of parameter names to refine (None = all continuous params)
        params_to_fix: Dict mapping param names to fixed values (removes from search space)
    """

    enabled: bool = False
    previous_phase_dir: str | None = None
    top_k: int = 20
    margin_factor: float = 0.3
    min_samples: int = 3
    use_percentiles: bool = True
    percentile_low: float = 10.0
    percentile_high: float = 90.0
    params_to_refine: list[str] | None = None
    params_to_fix: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinedRange:
    """Refined parameter range from previous phase analysis.

    Attributes:
        param_name: Name of the parameter
        original_type: Original parameter type (uniform, loguniform, choice, etc.)
        observed_values: List of observed values from top configs
        refined_low: Refined lower bound
        refined_high: Refined upper bound
        fixed_value: If not None, parameter should be fixed to this value
        refined_values: For choice params, refined list of values
    """

    param_name: str
    original_type: str
    observed_values: list[Any]
    refined_low: float | None = None
    refined_high: float | None = None
    fixed_value: Any | None = None
    refined_values: list[Any] | None = None


def load_top_configs(
    phase_dir: str | Path,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Load top configurations from a previous HPO phase.

    Attempts to load from:
    1. top_k_configs.json (preferred, already filtered)
    2. hpo_results.csv (fallback, needs filtering)

    Args:
        phase_dir: Directory containing HPO outputs
        top_k: Maximum number of configs to return

    Returns:
        List of configuration dictionaries

    Raises:
        FileNotFoundError: If no valid config source is found
    """
    phase_path = Path(phase_dir)

    # Try top_k_configs.json first
    top_k_path = phase_path / "top_k_configs.json"
    if top_k_path.exists():
        with open(top_k_path, encoding="utf-8") as f:
            configs = json.load(f)
        logger.info("Loaded %d configs from %s", len(configs), top_k_path)
        return configs[:top_k]

    # Try hpo_results.csv
    results_path = phase_path / "hpo_results.csv"
    if results_path.exists():
        import pandas as pd

        df = pd.read_csv(results_path)
        # Find metric column (val_mae, val_loss, etc.)
        metric_cols = [c for c in df.columns if c.startswith("val_")]
        if metric_cols:
            metric = metric_cols[0]
            df = df.sort_values(metric, ascending=True)  # Assume minimization

        # Extract config columns
        config_cols = [c for c in df.columns if c.startswith("config/")]
        configs = []
        for _, row in df.head(top_k).iterrows():
            config = {}
            for col in config_cols:
                param_name = col.replace("config/", "")
                value = row[col]
                if pd.notna(value):
                    config[param_name] = value
            configs.append(config)

        logger.info("Loaded %d configs from %s", len(configs), results_path)
        return configs

    # Try Optuna trials CSV
    optuna_path = phase_path / "optuna_trials.csv"
    if optuna_path.exists():
        import pandas as pd

        df = pd.read_csv(optuna_path)
        if "value" in df.columns:
            df = df.sort_values("value", ascending=True)

        # Extract params columns
        param_cols = [c for c in df.columns if c.startswith("params_")]
        configs = []
        for _, row in df.head(top_k).iterrows():
            config = {}
            for col in param_cols:
                param_name = col.replace("params_", "")
                value = row[col]
                if pd.notna(value):
                    config[param_name] = value
            configs.append(config)

        logger.info("Loaded %d configs from %s", len(configs), optuna_path)
        return configs

    raise FileNotFoundError(
        f"No valid config source found in {phase_dir}. "
        "Expected: top_k_configs.json, hpo_results.csv, or optuna_trials.csv"
    )


def _compute_refined_range(
    values: list[float | int],
    original_type: str,
    margin_factor: float = 0.3,
    use_percentiles: bool = True,
    percentile_low: float = 10.0,
    percentile_high: float = 90.0,
) -> tuple[float, float] | None:
    """Compute refined range from observed values.

    Args:
        values: List of observed values
        original_type: Parameter type (affects margin calculation)
        margin_factor: Factor to expand ranges (0.0-1.0)
        use_percentiles: Use percentile-based ranges
        percentile_low: Lower percentile
        percentile_high: Upper percentile

    Returns:
        Tuple of (low, high) for the refined range, or None if no valid values
    """
    import numpy as np

    values_arr = np.array([v for v in values if v is not None and not np.isnan(v)])

    if len(values_arr) == 0:
        return None  # Signal to caller to use original param unchanged

    # Compute base range
    if use_percentiles and len(values_arr) >= 5:
        low = np.percentile(values_arr, percentile_low)
        high = np.percentile(values_arr, percentile_high)
    else:
        low = float(np.min(values_arr))
        high = float(np.max(values_arr))

    # Ensure low < high (can happen with few samples)
    if low >= high:
        low = float(np.min(values_arr))
        high = float(np.max(values_arr))
        if low >= high:
            # All same value - expand slightly
            low = low * 0.9 if low > 0 else low - 0.1
            high = high * 1.1 if high > 0 else high + 0.1

    # Apply margin based on parameter type
    if original_type == "loguniform":
        # For log-scale params, apply margin in log space
        if low > 0 and high > 0:
            log_low = math.log(low)
            log_high = math.log(high)
            log_range = log_high - log_low
            log_margin = log_range * margin_factor

            refined_low = math.exp(log_low - log_margin)
            refined_high = math.exp(log_high + log_margin)
        else:
            # Fall back to linear margin for non-positive values
            value_range = high - low
            margin = value_range * margin_factor
            refined_low = low - margin
            refined_high = high + margin
    else:
        # Linear margin for uniform and other types
        value_range = high - low
        margin = value_range * margin_factor
        refined_low = low - margin
        refined_high = high + margin

    return float(refined_low), float(refined_high)


def analyze_parameter(
    param_name: str,
    configs: list[dict[str, Any]],
    original_param: ParameterSpace | None,
    margin_factor: float = 0.3,
    use_percentiles: bool = True,
    percentile_low: float = 10.0,
    percentile_high: float = 90.0,
) -> RefinedRange:
    """Analyze a parameter from top configs and compute refined range.

    Args:
        param_name: Name of the parameter
        configs: List of top configuration dictionaries
        original_param: Original ParameterSpace definition (for type info)
        margin_factor: Factor to expand ranges
        use_percentiles: Use percentile-based ranges
        percentile_low: Lower percentile
        percentile_high: Upper percentile

    Returns:
        RefinedRange with computed bounds
    """
    # Extract values for this parameter
    values = []
    for cfg in configs:
        if param_name in cfg:
            val = cfg[param_name]
            if val is not None:
                values.append(val)

    # Determine original type
    original_type = "uniform"
    if original_param is not None:
        original_type = original_param.type

    refined = RefinedRange(
        param_name=param_name,
        original_type=original_type,
        observed_values=values,
    )

    if not values:
        logger.warning("No values found for parameter: %s", param_name)
        return refined

    # Handle choice parameters
    if original_type == "choice":
        # Keep only values that appeared in top configs
        unique_values = list(set(values))
        refined.refined_values = sorted(unique_values, key=lambda x: (x is None, x))

        # If only one value, fix it
        if len(unique_values) == 1:
            refined.fixed_value = unique_values[0]
            logger.info("Parameter %s: fixing to %s (only value observed)", param_name, unique_values[0])
        else:
            logger.info("Parameter %s: narrowed choices to %s", param_name, unique_values)
        return refined

    # Handle continuous parameters
    if original_type in ("uniform", "loguniform", "quniform", "randint", "qrandint"):
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if numeric_values:
            result = _compute_refined_range(
                numeric_values,
                original_type,
                margin_factor,
                use_percentiles,
                percentile_low,
                percentile_high,
            )

            # If no valid values after NaN filtering, keep original
            if result is None:
                logger.warning(
                    "Parameter %s: no valid values after filtering, keeping original range",
                    param_name,
                )
                return refined

            refined.refined_low, refined.refined_high = result

            # For integer types, round bounds
            if original_type in ("randint", "qrandint"):
                refined.refined_low = int(math.floor(refined.refined_low))
                refined.refined_high = int(math.ceil(refined.refined_high))

            # For quantized types, round bounds to multiples of q
            if original_type in ("quniform", "qrandint") and original_param is not None:
                q = original_param.q
                if q is not None and q > 0:
                    # Round low down to nearest multiple of q
                    refined.refined_low = math.floor(refined.refined_low / q) * q
                    # Round high up to nearest multiple of q
                    refined.refined_high = math.ceil(refined.refined_high / q) * q

            logger.info(
                "Parameter %s: refined [%.6g, %.6g] (observed %d values)",
                param_name,
                refined.refined_low,
                refined.refined_high,
                len(numeric_values),
            )

    return refined


def refine_search_space(
    base_config: SearchSpaceConfig,
    refinement: RefinementConfig,
) -> SearchSpaceConfig:
    """Refine a search space configuration using results from a previous phase.

    This is the main entry point for automatic Phase 3 refinement. It:
    1. Loads top configurations from the previous phase
    2. Analyzes parameter distributions
    3. Returns an updated SearchSpaceConfig with narrowed ranges

    Args:
        base_config: Original SearchSpaceConfig (e.g., Phase 3 template)
        refinement: RefinementConfig specifying how to refine

    Returns:
        New SearchSpaceConfig with refined parameter ranges

    Example:
        >>> refinement = RefinementConfig(
        ...     enabled=True,
        ...     previous_phase_dir="/path/to/phase2",
        ...     top_k=20,
        ...     margin_factor=0.3,
        ... )
        >>> refined = refine_search_space(base_config, refinement)
    """
    if not refinement.enabled:
        logger.info("Refinement disabled, returning base config unchanged")
        return base_config

    if refinement.previous_phase_dir is None:
        logger.warning("previous_phase_dir not set, returning base config unchanged")
        return base_config

    # Load top configs from previous phase
    try:
        top_configs = load_top_configs(
            refinement.previous_phase_dir,
            top_k=refinement.top_k,
        )
    except FileNotFoundError as e:
        logger.error("Failed to load previous phase configs: %s", e)
        return base_config

    if len(top_configs) < refinement.min_samples:
        logger.warning(
            "Insufficient configs from previous phase (%d < %d), using base config",
            len(top_configs),
            refinement.min_samples,
        )
        return base_config

    logger.info(
        "Refining search space using %d top configs from %s",
        len(top_configs),
        refinement.previous_phase_dir,
    )

    # Identify parameters to refine
    all_params = _get_all_param_names(base_config)
    params_to_refine = set(refinement.params_to_refine or all_params)

    # Build refined parameter spaces
    refined_params: dict[str, ParameterSpace | None] = {}

    for param_name in all_params:
        original_param = getattr(base_config, param_name, None)

        # Check if parameter should be fixed
        if param_name in refinement.params_to_fix:
            # Mark for removal (will not be tuned)
            refined_params[param_name] = None
            logger.info("Parameter %s: fixed to %s", param_name, refinement.params_to_fix[param_name])
            continue

        # Skip if not in refinement list
        if param_name not in params_to_refine:
            refined_params[param_name] = original_param
            continue

        # Skip if original param not defined
        if original_param is None:
            continue

        # Analyze and refine
        refined_range = analyze_parameter(
            param_name=param_name,
            configs=top_configs,
            original_param=original_param,
            margin_factor=refinement.margin_factor,
            use_percentiles=refinement.use_percentiles,
            percentile_low=refinement.percentile_low,
            percentile_high=refinement.percentile_high,
        )

        # Create refined ParameterSpace
        refined_param = _create_refined_parameter_space(original_param, refined_range)
        refined_params[param_name] = refined_param

    # Build new SearchSpaceConfig
    return _build_refined_config(base_config, refined_params)


def _get_all_param_names(config: SearchSpaceConfig) -> list[str]:
    """Get all parameter names from a SearchSpaceConfig."""
    return [
        "learning_rate",
        "lr_warmup_ratio",
        "lr_final_ratio",
        "warmup_epochs",
        "patience",
        "dropout",
        "batch_norm",
        "weight_decay_enabled",
        "weight_decay",
        "depth",
        "message_hidden_dim",
        "ffn_num_layers",
        "ffn_hidden_dim",
        "batch_size",
        "ffn_type",
        "n_experts",
        "trunk_depth",
        "trunk_hidden_dim",
        "aggregation",
        "aggregation_norm",
        "target_weights",
    ]


def _create_refined_parameter_space(
    original: ParameterSpace,
    refined: RefinedRange,
) -> ParameterSpace:
    """Create a refined ParameterSpace from analysis results."""
    # If should be fixed, return None (handled separately)
    if refined.fixed_value is not None:
        # Return a choice with single value
        return ParameterSpace(
            type="choice",
            values=[refined.fixed_value],
            conditional_on=original.conditional_on,
            conditional_values=original.conditional_values,
        )

    # For choice parameters
    if original.type == "choice" and refined.refined_values is not None:
        return ParameterSpace(
            type="choice",
            values=refined.refined_values,
            conditional_on=original.conditional_on,
            conditional_values=original.conditional_values,
        )

    # For continuous parameters
    if refined.refined_low is not None and refined.refined_high is not None:
        # Ensure bounds respect original constraints
        low = refined.refined_low
        high = refined.refined_high

        # For loguniform, ensure positive bounds
        if original.type == "loguniform":
            low = max(low, 1e-10)
            if high <= low:
                high = low * 10

        return ParameterSpace(
            type=original.type,
            low=low,
            high=high,
            q=original.q,
            conditional_on=original.conditional_on,
            conditional_values=original.conditional_values,
        )

    # Fall back to original if no refinement possible
    return original


def _build_refined_config(
    base: SearchSpaceConfig,
    refined_params: dict[str, ParameterSpace | None],
) -> SearchSpaceConfig:
    """Build a new SearchSpaceConfig with refined parameters."""
    # Create kwargs for new config
    kwargs: dict[str, Any] = {}

    for param_name in _get_all_param_names(base):
        if param_name in refined_params:
            param = refined_params[param_name]
            if param is not None:  # None means parameter is fixed/removed
                kwargs[param_name] = param
        else:
            # Keep original
            original = getattr(base, param_name, None)
            if original is not None:
                kwargs[param_name] = original

    # Handle joint_sampling separately (nested config)
    if base.joint_sampling is not None:
        kwargs["joint_sampling"] = base.joint_sampling

    return SearchSpaceConfig(**kwargs)


def _get_chemeleon_param_names() -> list[str]:
    """Get all parameter names for ChemeleonSearchSpaceConfig."""
    return [
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
        "n_experts",
        "trunk_depth",
        "trunk_hidden_dim",
        "freeze_encoder",
        "unfreeze_encoder_epoch",
        "unfreeze_encoder_lr_multiplier",
    ]


def refine_chemeleon_search_space(
    base_config: Any,  # ChemeleonSearchSpaceConfig
    refinement: RefinementConfig,
) -> Any:
    """Refine a Chemeleon search space configuration using results from a previous phase.

    This is the Chemeleon-specific version of refine_search_space. It:
    1. Loads top configurations from the previous phase
    2. Analyzes parameter distributions
    3. Returns an updated ChemeleonSearchSpaceConfig with narrowed ranges

    Args:
        base_config: Original ChemeleonSearchSpaceConfig
        refinement: RefinementConfig specifying how to refine

    Returns:
        New ChemeleonSearchSpaceConfig with refined parameter ranges
    """
    # Import here to avoid circular imports
    from admet.model.chemeleon.hpo_config import ChemeleonSearchSpaceConfig, ParameterSpace as ChemeleonParameterSpace

    if not refinement.enabled:
        logger.info("Refinement disabled, returning base config unchanged")
        return base_config

    if refinement.previous_phase_dir is None:
        logger.warning("previous_phase_dir not set, returning base config unchanged")
        return base_config

    # Load top configs from previous phase
    try:
        top_configs = load_top_configs(
            refinement.previous_phase_dir,
            top_k=refinement.top_k,
        )
    except FileNotFoundError as e:
        logger.error("Failed to load previous phase configs: %s", e)
        return base_config

    if len(top_configs) < refinement.min_samples:
        logger.warning(
            "Insufficient configs from previous phase (%d < %d), using base config",
            len(top_configs),
            refinement.min_samples,
        )
        return base_config

    logger.info(
        "Refining Chemeleon search space using %d top configs from %s",
        len(top_configs),
        refinement.previous_phase_dir,
    )

    # Identify parameters to refine
    all_params = _get_chemeleon_param_names()
    params_to_refine = set(refinement.params_to_refine or all_params)

    # Build refined parameter spaces
    refined_params: dict[str, Any] = {}

    for param_name in all_params:
        original_param = getattr(base_config, param_name, None)

        # Check if parameter should be fixed
        if param_name in refinement.params_to_fix:
            refined_params[param_name] = None
            logger.info("Parameter %s: fixed to %s", param_name, refinement.params_to_fix[param_name])
            continue

        # Skip if not in refinement list
        if param_name not in params_to_refine:
            refined_params[param_name] = original_param
            continue

        # Skip if original param not defined
        if original_param is None:
            continue

        # Analyze and refine (use the generic analyze_parameter function)
        # Need to convert to ParameterSpace for analysis
        temp_param = ParameterSpace(
            type=original_param.type,
            low=original_param.low,
            high=original_param.high,
            values=original_param.values,
            q=original_param.q,
            conditional_on=original_param.conditional_on,
            conditional_values=original_param.conditional_values,
        )

        refined_range = analyze_parameter(
            param_name=param_name,
            configs=top_configs,
            original_param=temp_param,
            margin_factor=refinement.margin_factor,
            use_percentiles=refinement.use_percentiles,
            percentile_low=refinement.percentile_low,
            percentile_high=refinement.percentile_high,
        )

        # Create refined ChemeleonParameterSpace
        refined_param = _create_refined_chemeleon_param(original_param, refined_range, ChemeleonParameterSpace)
        refined_params[param_name] = refined_param

    # Build new ChemeleonSearchSpaceConfig
    kwargs: dict[str, Any] = {}
    for param_name in all_params:
        if param_name in refined_params:
            param = refined_params[param_name]
            if param is not None:
                kwargs[param_name] = param
        else:
            original = getattr(base_config, param_name, None)
            if original is not None:
                kwargs[param_name] = original

    # Handle joint_sampling separately
    if hasattr(base_config, "joint_sampling") and base_config.joint_sampling is not None:
        kwargs["joint_sampling"] = base_config.joint_sampling

    return ChemeleonSearchSpaceConfig(**kwargs)


def _create_refined_chemeleon_param(
    original: Any,
    refined: RefinedRange,
    param_class: type,
) -> Any:
    """Create a refined ParameterSpace for Chemeleon from analysis results."""
    if refined.fixed_value is not None:
        return param_class(
            type="choice",
            values=[refined.fixed_value],
            conditional_on=original.conditional_on,
            conditional_values=original.conditional_values,
        )

    if original.type == "choice" and refined.refined_values is not None:
        return param_class(
            type="choice",
            values=refined.refined_values,
            conditional_on=original.conditional_on,
            conditional_values=original.conditional_values,
        )

    if refined.refined_low is not None and refined.refined_high is not None:
        low = refined.refined_low
        high = refined.refined_high

        if original.type == "loguniform":
            low = max(low, 1e-10)
            if high <= low:
                high = low * 10

        return param_class(
            type=original.type,
            low=low,
            high=high,
            q=original.q,
            conditional_on=original.conditional_on,
            conditional_values=original.conditional_values,
        )

    return original


def generate_refined_phase3_config(
    phase2_dir: str | Path,
    phase3_template_path: str | Path | None = None,
    output_path: str | Path | None = None,
    top_k: int = 20,
    margin_factor: float = 0.3,
) -> dict[str, Any]:
    """Generate a refined Phase 3 config YAML from Phase 2 results.

    This is a convenience function that:
    1. Loads Phase 2 top configs
    2. Computes refined ranges
    3. Generates a complete Phase 3 config

    Args:
        phase2_dir: Directory containing Phase 2 outputs
        phase3_template_path: Optional path to Phase 3 template YAML
        output_path: Optional path to write refined config
        top_k: Number of top configs to analyze
        margin_factor: Margin factor for range expansion

    Returns:
        Dictionary containing the refined configuration
    """
    from omegaconf import OmegaConf

    # Load top configs
    top_configs = load_top_configs(phase2_dir, top_k=top_k)

    # Load template if provided
    if phase3_template_path is not None:
        template = OmegaConf.load(phase3_template_path)
        raw_dict = OmegaConf.to_container(template, resolve=True)
        # OmegaConf returns a complex union type, but we know it's a dict here
        template_dict: dict[str, Any] = dict(raw_dict) if isinstance(raw_dict, dict) else {}  # type: ignore[arg-type]
    else:
        template_dict = {}

    # Analyze all parameters
    search_space: dict[str, Any] = template_dict.get("search_space", {})
    refined_search_space = {}

    for param_name, param_config in search_space.items():
        if isinstance(param_config, dict) and "type" in param_config:
            # Analyze this parameter
            original_type = param_config.get("type", "uniform")
            values = [cfg.get(param_name) for cfg in top_configs if param_name in cfg]
            values = [v for v in values if v is not None]

            if not values:
                # Keep original config
                refined_search_space[param_name] = param_config
                continue

            # Compute refined range based on type
            if original_type == "choice":
                unique_values = list(set(values))
                if len(unique_values) == 1:
                    # Fix to single value
                    refined_search_space[param_name] = {
                        "type": "choice",
                        "values": unique_values,
                    }
                else:
                    refined_search_space[param_name] = {
                        "type": "choice",
                        "values": sorted(unique_values, key=lambda x: (x is None, str(x))),
                    }
            elif original_type in ("uniform", "loguniform", "quniform", "randint", "qrandint"):
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    low, high = _compute_refined_range(
                        numeric_values,
                        original_type,
                        margin_factor=margin_factor,
                    )

                    refined_config = {
                        "type": original_type,
                        "low": low,
                        "high": high,
                    }
                    if param_config.get("q") is not None:
                        refined_config["q"] = param_config["q"]
                    if param_config.get("conditional_on") is not None:
                        refined_config["conditional_on"] = param_config["conditional_on"]
                        refined_config["conditional_values"] = param_config.get("conditional_values")

                    refined_search_space[param_name] = refined_config
                else:
                    refined_search_space[param_name] = param_config
            else:
                refined_search_space[param_name] = param_config

            # Preserve conditionals
            if "conditional_on" in param_config:
                refined_search_space[param_name]["conditional_on"] = param_config["conditional_on"]
                refined_search_space[param_name]["conditional_values"] = param_config.get("conditional_values")
        else:
            # Nested config (like joint_sampling) - keep as-is
            refined_search_space[param_name] = param_config

    # Update template with refined search space
    result = dict(template_dict)
    result["search_space"] = refined_search_space

    # Add metadata about refinement
    result["_refinement_metadata"] = {
        "source_phase": str(phase2_dir),
        "top_k_analyzed": len(top_configs),
        "margin_factor": margin_factor,
    }

    # Write to file if output path provided
    if output_path is not None:
        import yaml  # type: ignore[import-untyped]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Add header comment
            f.write("# Auto-generated Phase 3 config with refined search ranges\n")
            f.write(f"# Source: {phase2_dir}\n")
            f.write(f"# Top-{len(top_configs)} configs analyzed\n\n")

            # Remove metadata before writing (it's just for programmatic use)
            output_dict = {k: v for k, v in result.items() if not k.startswith("_")}
            yaml.dump(output_dict, f, default_flow_style=False, sort_keys=False)

        logger.info("Wrote refined Phase 3 config to: %s", output_path)

    return result
