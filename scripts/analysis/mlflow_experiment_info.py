#!/usr/bin/env python3
"""
Find MLflow experiment names from IDs and show disk usage information.

This script helps identify which experiments are taking up disk space
and shows how many runs (active and deleted) exist in each experiment.
"""

import argparse
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def get_directory_size(path: Path) -> int:
    """Get size of directory in bytes using du command."""
    try:
        result = subprocess.run(
            ["/usr/bin/du", "-sb", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.split()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0


def format_bytes(bytes_size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def main():
    parser = argparse.ArgumentParser(description="Show MLflow experiment information and disk usage")
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
        help="Show details for specific experiment ID only",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Show individual runs (can be slow for large experiments)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["id", "name", "size", "runs", "deleted"],
        default="size",
        help="Sort experiments by (default: size)",
    )

    args = parser.parse_args()

    # Connect to MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    artifact_root = Path(args.artifact_root)

    print(f"MLflow Tracking URI: {args.tracking_uri}")
    print(f"Artifact Root: {artifact_root}")
    print("=" * 100)

    # Get all experiments
    experiments = client.search_experiments()

    # Filter if specific experiment requested
    if args.experiment_id:
        experiments = [exp for exp in experiments if exp.experiment_id == args.experiment_id]

    # Collect experiment data
    exp_data = []
    for exp in experiments:
        exp_id = exp.experiment_id
        exp_name = exp.name

        # Count active runs
        active_runs = client.search_runs(
            experiment_ids=[exp_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        )
        active_count = len(active_runs)

        # Count deleted runs
        deleted_runs = client.search_runs(
            experiment_ids=[exp_id],
            run_view_type=mlflow.entities.ViewType.DELETED_ONLY,
        )
        deleted_count = len(deleted_runs)

        # Get disk usage
        exp_artifact_path = artifact_root / exp_id
        if exp_artifact_path.exists():
            size_bytes = get_directory_size(exp_artifact_path)
            size_str = format_bytes(size_bytes)
        else:
            size_bytes = 0
            size_str = "0B"

        exp_data.append(
            {
                "id": exp_id,
                "name": exp_name,
                "active": active_count,
                "deleted": deleted_count,
                "size_bytes": size_bytes,
                "size_str": size_str,
                "path": exp_artifact_path,
            }
        )

    # Sort experiments
    sort_key = {
        "id": lambda x: int(x["id"]) if x["id"].isdigit() else x["id"],
        "name": lambda x: x["name"],
        "size": lambda x: x["size_bytes"],
        "runs": lambda x: x["active"],
        "deleted": lambda x: x["deleted"],
    }[args.sort_by]

    exp_data.sort(key=sort_key, reverse=(args.sort_by in ["size", "runs", "deleted"]))

    # Print experiment summary
    print(f"\n{'ID':<6} {'Name':<40} {'Active':<8} {'Deleted':<8} {'Disk Usage':<12} {'Path'}")
    print("-" * 100)

    total_size = 0
    total_active = 0
    total_deleted = 0

    for exp in exp_data:
        print(
            f"{exp['id']:<6} {exp['name']:<40} {exp['active']:<8} {exp['deleted']:<8} "
            f"{exp['size_str']:<12} {exp['path']}"
        )
        total_size += exp["size_bytes"]
        total_active += exp["active"]
        total_deleted += exp["deleted"]

    print("-" * 100)
    print(f"{'TOTAL':<6} {'':<40} {total_active:<8} {total_deleted:<8} " f"{format_bytes(total_size):<12}")

    # Show individual runs if requested
    if args.show_runs and exp_data:
        for exp in exp_data:
            if exp["deleted"] > 0:
                print(f"\n\nDeleted runs in experiment {exp['id']} ({exp['name']}):")
                print(f"{'Run ID':<40} {'Artifact URI'}")
                print("-" * 100)

                deleted_runs = client.search_runs(
                    experiment_ids=[exp["id"]],
                    run_view_type=mlflow.entities.ViewType.DELETED_ONLY,
                )

                for run in deleted_runs:
                    print(f"{run.info.run_id:<40} {run.info.artifact_uri}")

    # Print cleanup suggestions
    if total_deleted > 0:
        print("\n" + "=" * 100)
        print("CLEANUP SUGGESTIONS:")
        print("=" * 100)
        print(f"\nFound {total_deleted} deleted runs taking up {format_bytes(total_size)} of disk space.")
        print("\nTo delete artifacts for deleted runs in a specific experiment, run:")
        print("\n  python scripts/analysis/mlflow_cleanup.py --experiment-id <ID>")
        print("\nOr to delete an entire experiment's artifacts:")
        print("\n  rm -rf /media/aglisman/Data/models/mlflow-artifacts/<ID>")
        print("\nWarning: Make sure runs are marked as deleted in MLflow before removing artifacts!")


if __name__ == "__main__":
    main()
