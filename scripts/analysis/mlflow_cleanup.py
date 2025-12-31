#!/usr/bin/env python3
"""
Clean up MLflow artifacts for deleted runs.

This script removes artifact files for runs that have been marked as deleted
in the MLflow tracking server, freeing up disk space.
"""

import argparse
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def format_bytes(bytes_size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (PermissionError, FileNotFoundError):
        pass
    return total


def main():
    parser = argparse.ArgumentParser(description="Clean up MLflow artifacts for deleted runs")
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:8084",
        help="MLflow tracking URI (default: http://127.0.0.1:8084)",
    )
    parser.add_argument(
        "--artifact-root",
        default="/media/aglisman/Data/models/mlflow-artifacts",
        help="Path to MLflow artifacts directory",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to clean up",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Connect to MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    artifact_root = Path(args.artifact_root)
    exp_artifact_path = artifact_root / args.experiment_id

    if not exp_artifact_path.exists():
        print(f"Error: Experiment artifact path does not exist: {exp_artifact_path}")
        return 1

    print(f"MLflow Tracking URI: {args.tracking_uri}")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Artifact Path: {exp_artifact_path}")
    print("=" * 80)

    # Get experiment info
    try:
        experiment = client.get_experiment(args.experiment_id)
        print(f"Experiment Name: {experiment.name}")
    except Exception as e:
        print(f"Warning: Could not get experiment info: {e}")

    # Get deleted runs
    deleted_runs = client.search_runs(
        experiment_ids=[args.experiment_id],
        run_view_type=mlflow.entities.ViewType.DELETED_ONLY,
    )

    if not deleted_runs:
        print("\nNo deleted runs found. Nothing to clean up.")
        return 0

    print(f"\nFound {len(deleted_runs)} deleted runs")
    print("=" * 80)

    # Calculate total size to be freed
    total_size = 0
    runs_to_delete = []

    for run in deleted_runs:
        run_artifact_path = exp_artifact_path / run.info.run_id
        if run_artifact_path.exists():
            size = get_directory_size(run_artifact_path)
            total_size += size
            runs_to_delete.append(
                {
                    "run_id": run.info.run_id,
                    "path": run_artifact_path,
                    "size": size,
                }
            )
            print(f"  {run.info.run_id}: {format_bytes(size)}")

    if not runs_to_delete:
        print("\nNo artifact files found for deleted runs. Already cleaned up?")
        return 0

    print("=" * 80)
    print(f"Total space to be freed: {format_bytes(total_size)}")
    print(f"Number of run directories to delete: {len(runs_to_delete)}")

    if args.dry_run:
        print("\n[DRY RUN] No files were deleted. Remove --dry-run to actually delete.")
        return 0

    # Confirmation prompt
    if not args.force:
        print("\n⚠️  WARNING: This will permanently delete artifact files!")
        response = input("\nDo you want to proceed? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            return 0

    # Delete artifacts
    print("\nDeleting artifacts...")
    deleted_count = 0
    freed_space = 0

    for item in runs_to_delete:
        try:
            shutil.rmtree(item["path"])
            deleted_count += 1
            freed_space += item["size"]
            print(f"  ✓ Deleted {item['run_id']}")
        except Exception as e:
            print(f"  ✗ Failed to delete {item['run_id']}: {e}")

    print("=" * 80)
    print("Cleanup complete!")
    print(f"  Deleted: {deleted_count}/{len(runs_to_delete)} run directories")
    print(f"  Freed: {format_bytes(freed_space)}")

    return 0


if __name__ == "__main__":
    exit(main())
