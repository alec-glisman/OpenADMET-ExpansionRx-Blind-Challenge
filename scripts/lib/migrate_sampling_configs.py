#!/usr/bin/env python3
"""
Migrate YAML configuration files to new joint_sampling schema.

This script converts legacy configuration files that use:
- optimization.task_sampling_alpha
- curriculum.enabled

To the new unified schema:
- joint_sampling.enabled
- joint_sampling.task_oversampling.alpha
- joint_sampling.curriculum.enabled

Usage:
    python scripts/lib/migrate_sampling_configs.py --config-dir configs/0-experiment
    python scripts/lib/migrate_sampling_configs.py --config-file configs/0-experiment/chemprop.yaml
    python scripts/lib/migrate_sampling_configs.py --config-dir configs --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def migrate_config(config_path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single configuration file to new joint_sampling schema.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration file.
    dry_run : bool, default=False
        If True, print changes without modifying files.

    Returns
    -------
    bool
        True if migration was needed and performed, False otherwise.
    """
    try:
        config = OmegaConf.load(config_path)
    except Exception as e:
        logger.error("Failed to load %s: %s", config_path, e)
        return False

    migrated = False

    # Check if migration is needed
    has_task_sampling_alpha = (
        hasattr(config, "optimization")
        and hasattr(config.optimization, "task_sampling_alpha")
        and config.optimization.task_sampling_alpha is not None
    )

    has_curriculum_section = hasattr(config, "curriculum")
    has_curriculum_enabled = (
        has_curriculum_section and hasattr(config.curriculum, "enabled") and config.curriculum.enabled
    )

    # Check for search_space.task_sampling_alpha (HPO configs)
    has_search_space_task_sampling_alpha = hasattr(config, "search_space") and hasattr(
        config.search_space, "task_sampling_alpha"
    )

    # Check if already migrated
    has_joint_sampling = hasattr(config, "joint_sampling")

    if (
        has_joint_sampling
        and not has_task_sampling_alpha
        and not has_curriculum_section
        and not has_search_space_task_sampling_alpha
    ):
        logger.info("✓ %s already fully migrated", config_path.name)
        return False

    if not has_task_sampling_alpha and not has_curriculum_section and not has_search_space_task_sampling_alpha:
        logger.info("✓ %s needs no migration (no sampling config)", config_path.name)
        return False

    # Perform migration
    logger.info("⚙ Migrating %s", config_path.name)

    # Create joint_sampling section if needed
    if not has_joint_sampling:
        joint_sampling = OmegaConf.create(
            {
                "enabled": False,
                "task_oversampling": {"alpha": 0.0},
                "curriculum": {
                    "enabled": False,
                    "quality_col": "Quality",
                    "qualities": ["high", "medium", "low"],
                    "patience": 5,
                    "strategy": "sampled",
                    "reset_early_stopping_on_phase_change": False,
                    "log_per_quality_metrics": True,
                },
                "num_samples": None,
                "seed": 42,
                "increment_seed_per_epoch": True,
                "log_to_mlflow": True,
            }
        )
    else:
        joint_sampling = config.joint_sampling

    # Migrate task_sampling_alpha
    if has_task_sampling_alpha:
        alpha = config.optimization.task_sampling_alpha
        joint_sampling.enabled = True
        joint_sampling.task_oversampling.alpha = alpha
        logger.info("  • Migrated task_sampling_alpha=%.4f → joint_sampling.task_oversampling.alpha", alpha)
        migrated = True

    # Migrate curriculum config (including disabled ones for their settings)
    if has_curriculum_section:
        # Copy over curriculum settings to joint_sampling.curriculum
        if has_curriculum_enabled:
            joint_sampling.enabled = True
        # Merge curriculum settings
        for key in [
            "enabled",
            "quality_col",
            "qualities",
            "patience",
            "seed",
            "strategy",
            "reset_early_stopping_on_phase_change",
            "log_per_quality_metrics",
        ]:
            if hasattr(config.curriculum, key):
                joint_sampling.curriculum[key] = config.curriculum[key]
        logger.info(
            "  • Migrated curriculum settings → joint_sampling.curriculum (enabled=%s)",
            joint_sampling.curriculum.enabled,
        )
        migrated = True

    if migrated or has_task_sampling_alpha or has_curriculum_section:
        # Add/update joint_sampling in config
        OmegaConf.set_struct(config, False)
        config.joint_sampling = joint_sampling

        # Remove legacy task_sampling_alpha
        if has_task_sampling_alpha:
            del config.optimization.task_sampling_alpha  # type: ignore[union-attr]
            logger.info("  • Removed optimization.task_sampling_alpha")

        # Remove legacy curriculum section entirely
        if has_curriculum_section:
            del config.curriculum  # type: ignore[union-attr]
            logger.info("  • Removed legacy curriculum section")

    # Handle search_space.task_sampling_alpha migration (HPO configs)
    if has_search_space_task_sampling_alpha:
        OmegaConf.set_struct(config, False)

        # Get the old search space config
        old_alpha_config = config.search_space.task_sampling_alpha

        # Create joint_sampling search space with nested structure
        # Move task_sampling_alpha values to joint_sampling.task_oversampling.alpha
        config.search_space.joint_sampling = OmegaConf.create(
            {
                "enabled": {"type": "choice", "values": [True, False]},
                "task_oversampling": {"alpha": old_alpha_config},
            }
        )

        # Remove old task_sampling_alpha
        del config.search_space.task_sampling_alpha
        logger.info("  • Migrated search_space.task_sampling_alpha → search_space.joint_sampling")
        migrated = True

    # Save if any migration was performed
    if migrated:
        OmegaConf.set_struct(config, True)

        if not dry_run:
            # Save migrated config
            OmegaConf.save(config, config_path)
            logger.info("✅ Saved migrated config to %s", config_path)
        else:
            logger.info("🔍 DRY RUN: Would save to %s", config_path)
            if has_joint_sampling or has_task_sampling_alpha or has_curriculum_section:
                logger.info("\nMigrated joint_sampling config preview:")
                print(OmegaConf.to_yaml(config.joint_sampling))
            if has_search_space_task_sampling_alpha:
                logger.info("\nMigrated search_space.joint_sampling preview:")
                print(OmegaConf.to_yaml(config.search_space.joint_sampling))

    return migrated


def migrate_directory(config_dir: Path, dry_run: bool = False, recursive: bool = True) -> dict[str, int]:
    """
    Migrate all YAML files in a directory.

    Parameters
    ----------
    config_dir : Path
        Directory containing YAML configuration files.
    dry_run : bool, default=False
        If True, print changes without modifying files.
    recursive : bool, default=True
        If True, search subdirectories recursively.

    Returns
    -------
    dict[str, int]
        Summary statistics: {"migrated": N, "skipped": M, "errors": K}
    """
    stats = {"migrated": 0, "skipped": 0, "errors": 0}

    if recursive:
        yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))
    else:
        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    logger.info("Found %d YAML files in %s", len(yaml_files), config_dir)

    for config_path in sorted(yaml_files):
        try:
            if migrate_config(config_path, dry_run=dry_run):
                stats["migrated"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            logger.error("Error migrating %s: %s", config_path, e)
            stats["errors"] += 1

    return stats


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate YAML config files to new joint_sampling schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Single config file to migrate",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Directory containing config files to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Validate arguments
    if not args.config_file and not args.config_dir:
        parser.error("Must specify either --config-file or --config-dir")

    if args.config_file and args.config_dir:
        parser.error("Cannot specify both --config-file and --config-dir")

    # Run migration
    if args.config_file:
        if not args.config_file.exists():
            logger.error("Config file not found: %s", args.config_file)
            return 1

        success = migrate_config(args.config_file, dry_run=args.dry_run)
        return 0 if success or args.dry_run else 1

    elif args.config_dir:
        if not args.config_dir.exists():
            logger.error("Config directory not found: %s", args.config_dir)
            return 1

        stats = migrate_directory(
            args.config_dir,
            dry_run=args.dry_run,
            recursive=not args.no_recursive,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Migration Summary:")
        logger.info("  Migrated: %d", stats["migrated"])
        logger.info("  Skipped:  %d", stats["skipped"])
        logger.info("  Errors:   %d", stats["errors"])
        logger.info("=" * 60)

        return 0 if stats["errors"] == 0 else 1

    return 0


if __name__ == "__main__":
    exit(main())
