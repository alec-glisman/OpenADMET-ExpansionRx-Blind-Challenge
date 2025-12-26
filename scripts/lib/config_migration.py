#!/usr/bin/env python
"""Config migration utilities for multi-model support.

This module provides utilities to migrate existing YAML configs to the new
multi-model config structure with `model.type` discriminator.

The new structure nests model-specific parameters under `model.<type>` and
adds a `model.type` field to specify which model to use.

Example:
    Old structure:
        model:
          depth: 5
          hidden_dim: 300

    New structure:
        model:
          type: chemprop
          chemprop:
            depth: 5
            hidden_dim: 300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def is_migrated(config: dict[str, Any]) -> bool:
    """Check if config is already migrated to new structure.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    bool
        True if config has model.type field, False otherwise.
    """
    model_section = config.get("model", {})
    return isinstance(model_section, dict) and "type" in model_section


def migrate_to_new_structure(
    config: dict[str, Any],
    model_type: str = "chemprop",
) -> dict[str, Any]:
    """Migrate config from old to new multi-model structure.

    Parameters
    ----------
    config : dict
        Original configuration dictionary.
    model_type : str, optional
        Model type to assign. Defaults to "chemprop".

    Returns
    -------
    dict
        Migrated configuration with model.type field.

    Examples
    --------
    >>> old_config = {"model": {"depth": 5}, "data": {"train_path": "train.csv"}}
    >>> new_config = migrate_to_new_structure(old_config)
    >>> new_config["model"]["type"]
    'chemprop'
    >>> new_config["model"]["chemprop"]["depth"]
    5
    """
    if is_migrated(config):
        logger.info("Config already migrated, skipping")
        return config

    # Deep copy to avoid modifying original
    new_config = dict(config)

    # Extract model section
    model_section = new_config.pop("model", {})

    # Create new nested structure
    new_config["model"] = {
        "type": model_type,
        model_type: model_section,
    }

    return new_config


def migrate_config_file(
    config_path: Path,
    model_type: str = "chemprop",
    dry_run: bool = False,
) -> bool:
    """Migrate a single config file in place.

    Parameters
    ----------
    config_path : Path
        Path to YAML config file.
    model_type : str, optional
        Model type to assign. Defaults to "chemprop".
    dry_run : bool, optional
        If True, don't write changes. Defaults to False.

    Returns
    -------
    bool
        True if file was migrated, False if already migrated or error.
    """
    try:
        # Use ruamel.yaml to preserve comments if available
        try:
            import ruamel.yaml

            yaml = ruamel.yaml.YAML()
            yaml.preserve_quotes = True

            with open(config_path) as f:
                config = yaml.load(f)

            if is_migrated(config):
                logger.info(f"Already migrated: {config_path}")
                return False

            migrated = migrate_to_new_structure(dict(config), model_type)

            if not dry_run:
                with open(config_path, "w") as f:
                    yaml.dump(migrated, f)
                logger.info(f"Migrated: {config_path}")
            else:
                logger.info(f"Would migrate: {config_path}")

            return True

        except ImportError:
            # Fall back to OmegaConf
            config = OmegaConf.load(config_path)
            config_dict = OmegaConf.to_container(config, resolve=True)

            if is_migrated(config_dict):
                logger.info(f"Already migrated: {config_path}")
                return False

            migrated = migrate_to_new_structure(config_dict, model_type)

            if not dry_run:
                OmegaConf.save(OmegaConf.create(migrated), config_path)
                logger.info(f"Migrated: {config_path}")
            else:
                logger.info(f"Would migrate: {config_path}")

            return True

    except Exception as e:
        logger.error(f"Error migrating {config_path}: {e}")
        return False


def migrate_config_directory(
    config_dir: Path,
    model_type: str = "chemprop",
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate all YAML configs in a directory.

    Parameters
    ----------
    config_dir : Path
        Directory containing YAML config files.
    model_type : str, optional
        Model type to assign. Defaults to "chemprop".
    dry_run : bool, optional
        If True, don't write changes. Defaults to False.

    Returns
    -------
    tuple[int, int]
        Tuple of (migrated_count, skipped_count).
    """
    migrated = 0
    skipped = 0

    for yaml_file in config_dir.rglob("*.yaml"):
        if migrate_config_file(yaml_file, model_type, dry_run):
            migrated += 1
        else:
            skipped += 1

    return migrated, skipped


def ensure_model_type(config: DictConfig, default_type: str = "chemprop") -> DictConfig:
    """Ensure config has model.type field, adding it if needed.

    This is for backward compatibility - existing configs without model.type
    will be assigned the default type.

    Parameters
    ----------
    config : DictConfig
        Configuration object.
    default_type : str, optional
        Default model type if not specified. Defaults to "chemprop".

    Returns
    -------
    DictConfig
        Configuration with model.type field.
    """
    if "model" not in config:
        config.model = {}

    if "type" not in config.model:
        # Old-style config - assume default type and nest params
        model_params = dict(config.model)
        config.model = OmegaConf.create(
            {
                "type": default_type,
                default_type: model_params,
            }
        )

    return config


def get_model_type(config: DictConfig) -> str:
    """Get model type from config, with backward compatibility.

    Parameters
    ----------
    config : DictConfig
        Configuration object.

    Returns
    -------
    str
        Model type string.
    """
    if "model" in config and "type" in config.model:
        return config.model.type
    return "chemprop"  # Default for legacy configs


def get_model_params(config: DictConfig, model_type: str | None = None) -> DictConfig:
    """Get model-specific parameters from config.

    Handles both old and new config structures.

    Parameters
    ----------
    config : DictConfig
        Configuration object.
    model_type : str, optional
        Model type. If None, determined from config.

    Returns
    -------
    DictConfig
        Model-specific parameters.
    """
    if model_type is None:
        model_type = get_model_type(config)

    # New structure: model.type and model.<type>
    if "type" in config.model:
        return config.model.get(model_type, OmegaConf.create({}))

    # Old structure: model contains params directly
    return config.model


def main():
    """CLI entry point for config migration."""
    parser = argparse.ArgumentParser(description="Migrate configs to multi-model structure")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing YAML configs",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="chemprop",
        help="Model type to assign (default: chemprop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just show what would be done",
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

    migrated, skipped = migrate_config_directory(
        args.config_dir,
        args.model_type,
        args.dry_run,
    )

    print(f"\nMigration complete: {migrated} migrated, {skipped} skipped")


if __name__ == "__main__":
    main()
