#!/usr/bin/env python3
"""
Migrate YAML configuration files to new ray config schema.

This script converts legacy configuration files that use:
- max_parallel
- ray_num_cpus
- ray_num_gpus

To the new unified schema:
- ray.max_parallel
- ray.num_cpus
- ray.num_gpus

Usage:
    python scripts/lib/migrate_ray_configs.py --config-dir configs/0-experiment
    python scripts/lib/migrate_ray_configs.py --config-file configs/0-experiment/chemprop.yaml
    python scripts/lib/migrate_ray_configs.py --config-dir configs --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def migrate_config(config_path: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single configuration file to new ray config schema.

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
    has_max_parallel = hasattr(config, "max_parallel") and config.max_parallel is not None
    has_ray_num_cpus = hasattr(config, "ray_num_cpus")
    has_ray_num_gpus = hasattr(config, "ray_num_gpus")

    # Check if already migrated
    has_ray_section = hasattr(config, "ray")

    if has_ray_section and not has_max_parallel and not has_ray_num_cpus and not has_ray_num_gpus:
        logger.info("✓ %s already fully migrated", config_path.name)
        return False

    if not has_max_parallel and not has_ray_num_cpus and not has_ray_num_gpus:
        logger.info("✓ %s needs no migration (no ray config)", config_path.name)
        return False

    # Perform migration
    logger.info("⚙ Migrating %s", config_path.name)

    # Create ray section if needed
    if not has_ray_section:
        ray_config = OmegaConf.create(
            {
                "max_parallel": 1,
                "num_cpus": None,
                "num_gpus": None,
            }
        )
    else:
        ray_config = config.ray

    # Migrate fields
    if has_max_parallel:
        ray_config.max_parallel = config.max_parallel
        logger.info("  • Migrated max_parallel=%d → ray.max_parallel", config.max_parallel)
        migrated = True

    if has_ray_num_cpus:
        ray_config.num_cpus = config.ray_num_cpus
        logger.info("  • Migrated ray_num_cpus=%s → ray.num_cpus", config.ray_num_cpus)
        migrated = True

    if has_ray_num_gpus:
        ray_config.num_gpus = config.ray_num_gpus
        logger.info("  • Migrated ray_num_gpus=%s → ray.num_gpus", config.ray_num_gpus)
        migrated = True

    if migrated:
        # Add/update ray in config
        OmegaConf.set_struct(config, False)
        config.ray = ray_config

        # Remove legacy fields
        if has_max_parallel:
            del config.max_parallel
            logger.info("  • Removed max_parallel")

        if has_ray_num_cpus:
            del config.ray_num_cpus
            logger.info("  • Removed ray_num_cpus")

        if has_ray_num_gpus:
            del config.ray_num_gpus
            logger.info("  • Removed ray_num_gpus")

        OmegaConf.set_struct(config, True)

        if not dry_run:
            # Save migrated config
            OmegaConf.save(config, config_path)
            logger.info("✅ Saved migrated config to %s", config_path)
        else:
            logger.info("🔍 DRY RUN: Would save to %s", config_path)
            logger.info("\nMigrated ray config preview:")
            print(OmegaConf.to_yaml(config.ray))

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
        description="Migrate YAML config files to new ray schema",
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
