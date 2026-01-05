#!/usr/bin/env python3
"""
Batch update script to add logging section to all YAML config files.

This script recursively scans all YAML files in configs/ and adds the
logging section if it doesn't already exist. Preserves file structure and
formatting as much as possible.

Usage:
    python scripts/add_logging_to_configs.py
    python scripts/add_logging_to_configs.py --dry-run
    python scripts/add_logging_to_configs.py --verbose
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Default logging configuration to add
DEFAULT_LOGGING_CONFIG = {
    "enabled": True,
    "verbose": 0,
    "max_total_logs_gb": 1.0,
    "fail_on_upload_error": True,
}


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for script execution."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file safely."""
    try:
        with open(file_path, "r") as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except yaml.YAMLError as e:
        logging.error(f"Failed to parse YAML {file_path}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return {}


def save_yaml(file_path: Path, content: Dict[str, Any], dry_run: bool = False) -> bool:
    """Save YAML file safely, preserving structure."""
    if dry_run:
        logging.info(f"[DRY RUN] Would save {file_path}")
        return True

    try:
        with open(file_path, "w") as f:
            yaml.dump(
                content,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        logging.info(f"Updated {file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to write {file_path}: {e}")
        return False


def has_logging_section(config: Dict[str, Any]) -> bool:
    """Check if config already has logging section."""
    return "logging" in config


def add_logging_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Add logging section to config if not present."""
    if not has_logging_section(config):
        config["logging"] = DEFAULT_LOGGING_CONFIG.copy()
        return True
    return False


def process_yaml_file(file_path: Path, dry_run: bool = False) -> bool:
    """Process a single YAML file to add logging section."""
    logging.debug(f"Processing {file_path}")

    # Load config
    config = load_yaml(file_path)
    if not config:
        logging.warning(f"Skipping empty file: {file_path}")
        return False

    # Check if already has logging section
    if has_logging_section(config):
        logging.debug(f"Already has logging section: {file_path}")
        return False

    # Add logging section
    was_modified = add_logging_section(config)

    if was_modified:
        logging.info(f"Adding logging section to {file_path}")
        return save_yaml(file_path, config, dry_run=dry_run)

    return False


def find_yaml_files(config_dir: Path) -> List[Path]:
    """Find all YAML files in config directory."""
    yaml_files = []

    for pattern in ["**/*.yaml", "**/*.yml"]:
        yaml_files.extend(config_dir.glob(pattern))

    return sorted(yaml_files)


def batch_update_configs(
    config_dir: Path = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Batch update all YAML files in config directory.

    Parameters
    ----------
    config_dir : Path, optional
        Configuration directory to scan. Defaults to project/configs/
    dry_run : bool, default False
        If True, don't actually modify files
    verbose : bool, default False
        Enable verbose logging

    Returns
    -------
    Dict[str, int]
        Statistics: {'total': int, 'updated': int, 'skipped': int, 'failed': int}
    """
    setup_logging(verbose=verbose)
    logger = logging.getLogger(__name__)

    # Default to project configs directory
    if config_dir is None:
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "configs"

    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        return {"total": 0, "updated": 0, "skipped": 0, "failed": 0}

    logger.info(f"Scanning config directory: {config_dir}")

    # Find all YAML files
    yaml_files = find_yaml_files(config_dir)
    logger.info(f"Found {len(yaml_files)} YAML files")

    if not yaml_files:
        logger.warning("No YAML files found")
        return {"total": 0, "updated": 0, "skipped": 0, "failed": 0}

    # Process each file
    stats = {"total": 0, "updated": 0, "skipped": 0, "failed": 0}

    for yaml_file in yaml_files:
        stats["total"] += 1
        try:
            was_updated = process_yaml_file(yaml_file, dry_run=dry_run)
            if was_updated:
                stats["updated"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            logger.error(f"Failed to process {yaml_file}: {e}")
            stats["failed"] += 1

    return stats


def print_summary(stats: Dict[str, int], dry_run: bool = False) -> None:
    """Print summary statistics."""
    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Summary:")
    print(f"  Total files scanned:   {stats['total']}")
    print(f"  Files updated:         {stats['updated']}")
    print(f"  Files skipped (already have logging): {stats['skipped']}")
    print(f"  Files failed:          {stats['failed']}")

    if stats["updated"] > 0:
        print("\nLogging configuration added:")
        for key, value in DEFAULT_LOGGING_CONFIG.items():
            print(f"  - {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch update YAML configs to add logging section",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show what would be updated (dry run)
    python scripts/add_logging_to_configs.py --dry-run

    # Actually update all configs
    python scripts/add_logging_to_configs.py

    # Update with verbose logging
    python scripts/add_logging_to_configs.py --verbose

    # Update specific directory
    python scripts/add_logging_to_configs.py --config-dir ./configs/0-experiment
        """,
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory to scan (default: project/configs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Run batch update
    stats = batch_update_configs(
        config_dir=args.config_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Print summary
    print_summary(stats, dry_run=args.dry_run)

    # Exit with success code
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
