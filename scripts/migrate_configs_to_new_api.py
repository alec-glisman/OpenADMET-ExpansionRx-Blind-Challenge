#!/usr/bin/env python3
"""
Migrate YAML config files to new multi-model API structure.

This script updates config files from the old format:
    model:
        depth: 5
        message_hidden_dim: 600
        ...

To the new format:
    model:
        type: chemprop
        chemprop:
            depth: 5
            message_hidden_dim: 600
            ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def migrate_config_file(file_path: Path, *, dry_run: bool = False) -> bool:
    """Migrate a single config file to the new API structure."""
    with open(file_path) as f:
        content = f.read()

    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"  ERROR: Failed to parse {file_path}: {e}")
        return False

    if config is None:
        print(f"  SKIP: {file_path} is empty")
        return False

    if "model" not in config:
        print(f"  SKIP: {file_path} has no model section")
        return False

    model_config = config["model"]

    # Already migrated if has 'type' key
    if isinstance(model_config, dict) and "type" in model_config:
        print(f"  SKIP: {file_path} already migrated")
        return False

    # Check if this looks like a chemprop config (has depth or message_hidden_dim)
    chemprop_indicators = ["depth", "message_hidden_dim", "ffn_type", "aggregation"]
    if not any(key in model_config for key in chemprop_indicators):
        print(f"  SKIP: {file_path} doesn't appear to be a chemprop config")
        return False

    # Migrate: nest model params under model.chemprop and add model.type
    new_model_config = {"type": "chemprop", "chemprop": model_config}
    config["model"] = new_model_config

    if dry_run:
        print(f"  DRY-RUN: Would migrate {file_path}")
        return True

    # Write back with preserved formatting where possible
    with open(file_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  MIGRATED: {file_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate config files to new multi-model API")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("configs")],
        help="Paths to config files or directories (default: configs/)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be changed without making changes")
    args = parser.parse_args()

    files_to_process: list[Path] = []

    for path in args.paths:
        if path.is_file() and path.suffix in {".yaml", ".yml"}:
            files_to_process.append(path)
        elif path.is_dir():
            files_to_process.extend(path.rglob("*.yaml"))
            files_to_process.extend(path.rglob("*.yml"))
        else:
            print(f"WARNING: {path} not found or not a valid path")

    if not files_to_process:
        print("No config files found to process")
        return

    print(f"Processing {len(files_to_process)} config file(s)...")

    migrated_count = 0
    for file_path in sorted(files_to_process):
        if migrate_config_file(file_path, dry_run=args.dry_run):
            migrated_count += 1

    print(f"\nMigration complete: {migrated_count} file(s) {'would be ' if args.dry_run else ''}migrated")


if __name__ == "__main__":
    main()
