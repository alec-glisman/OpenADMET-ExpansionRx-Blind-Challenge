#!/usr/bin/env python
"""Config validation utilities for multi-model support.

This module validates YAML configs against the expected schema for each model type.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from scripts.lib.config_migration import get_model_type

logger = logging.getLogger(__name__)


# Required fields per model type
REQUIRED_FIELDS: dict[str, list[str]] = {
    "chemprop": ["data", "model", "optimization"],
    "chemeleon": ["data", "model"],
    "xgboost": ["data", "model"],
    "lightgbm": ["data", "model"],
    "catboost": ["data", "model"],
}


def validate_config(config: DictConfig) -> tuple[bool, list[str]]:
    """Validate a configuration.

    Parameters
    ----------
    config : DictConfig
        Configuration to validate.

    Returns
    -------
    tuple[bool, list[str]]
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    # Check model type
    model_type = get_model_type(config)
    if model_type not in REQUIRED_FIELDS:
        errors.append(f"Unknown model type: {model_type}")
        return False, errors

    # Check required fields
    for field in REQUIRED_FIELDS[model_type]:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Model-specific validation
    if model_type == "chemprop":
        errors.extend(validate_chemprop_config(config))
    elif model_type == "chemeleon":
        errors.extend(validate_chemeleon_config(config))
    elif model_type in ["xgboost", "lightgbm", "catboost"]:
        errors.extend(validate_classical_config(config, model_type))

    return len(errors) == 0, errors


def validate_chemprop_config(config: DictConfig) -> list[str]:
    """Validate chemprop-specific configuration.

    Parameters
    ----------
    config : DictConfig
        Configuration to validate.

    Returns
    -------
    list[str]
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Data validation
    data = config.get("data", {})
    if "data_dir" not in data and "train_file" not in data:
        errors.append("data: must have either data_dir or train_file")

    if "target_cols" not in data or not data.target_cols:
        errors.append("data: target_cols must be specified")

    # Model validation
    model = config.get("model", {})
    if "depth" in model and model.depth < 1:
        errors.append("model: depth must be >= 1")
    if "dropout" in model and not (0 <= model.dropout <= 1):
        errors.append("model: dropout must be between 0 and 1")

    # Optimization validation
    opt = config.get("optimization", {})
    if "max_epochs" in opt and opt.max_epochs < 1:
        errors.append("optimization: max_epochs must be >= 1")
    if "batch_size" in opt and opt.batch_size < 1:
        errors.append("optimization: batch_size must be >= 1")

    return errors


def validate_chemeleon_config(config: DictConfig) -> list[str]:
    """Validate chemeleon-specific configuration.

    Parameters
    ----------
    config : DictConfig
        Configuration to validate.

    Returns
    -------
    list[str]
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Data validation
    data = config.get("data", {})
    if "target_cols" not in data or not data.target_cols:
        errors.append("data: target_cols must be specified")

    # Model validation
    model = config.get("model", {})
    chemeleon = model.get("chemeleon", model)

    if "freeze_encoder" in chemeleon and not isinstance(chemeleon.freeze_encoder, bool):
        errors.append("model.chemeleon: freeze_encoder must be boolean")

    return errors


def validate_classical_config(config: DictConfig, model_type: str) -> list[str]:
    """Validate classical model configuration (xgboost, lightgbm, catboost).

    Parameters
    ----------
    config : DictConfig
        Configuration to validate.
    model_type : str
        Model type being validated.

    Returns
    -------
    list[str]
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    # Data validation
    data = config.get("data", {})
    if "target_cols" not in data or not data.target_cols:
        errors.append("data: target_cols must be specified")

    # Fingerprint config validation
    model = config.get("model", {})
    fingerprint = model.get("fingerprint", {})
    fp_type = fingerprint.get("type", "morgan")

    if fp_type not in ["morgan", "rdkit", "maccs", "mordred"]:
        errors.append(f"model.fingerprint: unknown type {fp_type}")

    return errors


def validate_config_file(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a single config file.

    Parameters
    ----------
    config_path : Path
        Path to YAML config file.

    Returns
    -------
    tuple[bool, list[str]]
        Tuple of (is_valid, list of error messages).
    """
    try:
        config = OmegaConf.load(config_path)
        return validate_config(config)
    except Exception as e:
        return False, [f"Error loading config: {e}"]


def validate_config_directory(config_dir: Path) -> dict[Path, tuple[bool, list[str]]]:
    """Validate all configs in a directory.

    Parameters
    ----------
    config_dir : Path
        Directory containing YAML config files.

    Returns
    -------
    dict[Path, tuple[bool, list[str]]]
        Dictionary mapping file paths to (is_valid, errors) tuples.
    """
    results: dict[Path, tuple[bool, list[str]]] = {}

    for yaml_file in config_dir.rglob("*.yaml"):
        results[yaml_file] = validate_config_file(yaml_file)

    return results


def main():
    """CLI entry point for config validation."""
    parser = argparse.ArgumentParser(description="Validate YAML configs")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Single config file to validate",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.config_file:
        is_valid, errors = validate_config_file(args.config_file)
        if is_valid:
            print(f"✓ {args.config_file}: Valid")
        else:
            print(f"✗ {args.config_file}: Invalid")
            for error in errors:
                print(f"  - {error}")
        return 0 if is_valid else 1

    results = validate_config_directory(args.config_dir)

    valid_count = 0
    invalid_count = 0

    for path, (is_valid, errors) in sorted(results.items()):
        if is_valid:
            print(f"✓ {path}")
            valid_count += 1
        else:
            print(f"✗ {path}")
            for error in errors:
                print(f"  - {error}")
            invalid_count += 1

    print(f"\nValidation complete: {valid_count} valid, {invalid_count} invalid")
    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    exit(main())
