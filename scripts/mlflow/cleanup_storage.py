#!/usr/bin/env python3
"""Storage cleanup utilities for ML experiments.

This script helps reduce storage usage by:
1. Removing unnecessary prediction CSV files
2. Compressing existing model artifacts
3. Cleaning up old MLflow runs
4. Archiving or deleting old experiments

Usage:
    python scripts/mlflow/cleanup_storage.py --analyze          # Show storage analysis
    python scripts/mlflow/cleanup_storage.py --clean-predictions  # Remove training predictions
    python scripts/mlflow/cleanup_storage.py --compress-models   # Compress existing models
    python scripts/mlflow/cleanup_storage.py --clean-mlruns     # Clean old MLflow runs
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_size_mb(path: Path) -> float:
    """Get size of file or directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def format_size(size_mb: float) -> str:
    """Format size for display."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.2f} MB"


def analyze_storage(base_path: Path) -> dict:
    """Analyze storage usage."""
    models_path = base_path / "assets" / "models"
    mlruns_path = base_path / "mlruns"

    analysis = {
        "models_total": 0.0,
        "predictions_csv": 0.0,
        "archive": 0.0,
        "mlruns": 0.0,
        "model_files": 0.0,
    }

    if models_path.exists():
        analysis["models_total"] = get_size_mb(models_path)

        # Count prediction CSVs
        for csv_file in models_path.rglob("*predictions*.csv"):
            analysis["predictions_csv"] += get_size_mb(csv_file)

        # Count archive
        archive_path = models_path / "archive"
        if archive_path.exists():
            analysis["archive"] = get_size_mb(archive_path)

        # Count model files
        for ext in ("*.pkl", "*.joblib", "*.pt", "*.ckpt", "*.json"):
            for f in models_path.rglob(ext):
                if "archive" not in str(f):
                    analysis["model_files"] += get_size_mb(f)

    if mlruns_path.exists():
        analysis["mlruns"] = get_size_mb(mlruns_path)

    return analysis


def print_analysis(analysis: dict) -> None:
    """Print storage analysis."""
    print("\n" + "=" * 60)
    print("STORAGE ANALYSIS")
    print("=" * 60)
    print(f"Total models directory:     {format_size(analysis['models_total'])}")
    print(f"  - Prediction CSV files:   {format_size(analysis['predictions_csv'])}")
    print(f"  - Archive directory:      {format_size(analysis['archive'])}")
    print(f"  - Model files:            {format_size(analysis['model_files'])}")
    print(f"MLflow runs directory:      {format_size(analysis['mlruns'])}")
    print("=" * 60)

    potential_savings = analysis["predictions_csv"] + analysis["archive"]
    print(f"\nPotential savings (predictions + archive): {format_size(potential_savings)}")
    print("\nRecommendations:")
    if analysis["archive"] > 100:
        print("  - Delete or compress assets/models/archive/ to save space")
    if analysis["predictions_csv"] > 100:
        print("  - Remove training prediction CSVs with --clean-predictions")
    print("  - Enable compress_artifacts=true in config for new experiments")


def clean_predictions(base_path: Path, dry_run: bool = True) -> None:
    """Remove training prediction CSV files."""
    models_path = base_path / "assets" / "models"
    if not models_path.exists():
        logger.info("No models directory found")
        return

    removed_size = 0.0
    removed_count = 0

    for csv_file in models_path.rglob("*predictions*.csv"):
        # Only remove training predictions, keep test/validation
        if "/train/" in str(csv_file) or csv_file.parent.name == "train":
            size = get_size_mb(csv_file)
            if dry_run:
                logger.info("Would remove: %s (%s)", csv_file, format_size(size))
            else:
                csv_file.unlink()
                logger.info("Removed: %s (%s)", csv_file, format_size(size))
            removed_size += size
            removed_count += 1

    action = "Would remove" if dry_run else "Removed"
    logger.info("%s %d files, total %s", action, removed_count, format_size(removed_size))
    if dry_run:
        logger.info("Run with --execute to actually remove files")


def compress_models(base_path: Path, dry_run: bool = True) -> None:
    """Compress existing uncompressed model pickle files."""
    import joblib

    models_path = base_path / "assets" / "models"
    if not models_path.exists():
        logger.info("No models directory found")
        return

    compressed_count = 0
    saved_size = 0.0

    for pkl_file in models_path.rglob("*.pkl"):
        if "archive" in str(pkl_file):
            continue

        original_size = get_size_mb(pkl_file)

        # Check if already compressed by attempting to load and checking
        try:
            data = joblib.load(pkl_file)
        except Exception as e:
            logger.warning("Could not load %s: %s", pkl_file, e)
            continue

        if dry_run:
            # Estimate compression ratio (typically 50-80% reduction)
            estimated_savings = original_size * 0.6
            logger.info("Would compress: %s (est. save %s)", pkl_file, format_size(estimated_savings))
            saved_size += estimated_savings
            compressed_count += 1
        else:
            # Save with compression
            temp_file = pkl_file.with_suffix(".pkl.tmp")
            joblib.dump(data, temp_file, compress=3)

            new_size = get_size_mb(temp_file)
            if new_size < original_size:
                temp_file.rename(pkl_file)
                savings = original_size - new_size
                saved_size += savings
                compressed_count += 1
                logger.info("Compressed: %s (%s -> %s)", pkl_file, format_size(original_size), format_size(new_size))
            else:
                temp_file.unlink()
                logger.info("Skipped (already optimal): %s", pkl_file)

    action = "Would compress" if dry_run else "Compressed"
    logger.info("%s %d files, total savings %s", action, compressed_count, format_size(saved_size))
    if dry_run:
        logger.info("Run with --execute to actually compress files")


def clean_mlruns(base_path: Path, keep_days: int = 30, dry_run: bool = True) -> None:
    """Clean old MLflow runs."""
    import time

    mlruns_path = base_path / "mlruns"
    if not mlruns_path.exists():
        logger.info("No mlruns directory found")
        return

    cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
    removed_size = 0.0
    removed_count = 0

    for experiment_dir in mlruns_path.iterdir():
        if not experiment_dir.is_dir():
            continue
        if experiment_dir.name in ("0", ".trash"):
            continue

        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue

            # Check modification time
            try:
                mtime = run_dir.stat().st_mtime
                if mtime < cutoff_time:
                    size = get_size_mb(run_dir)
                    if dry_run:
                        logger.info("Would remove: %s (%s)", run_dir, format_size(size))
                    else:
                        shutil.rmtree(run_dir)
                        logger.info("Removed: %s (%s)", run_dir, format_size(size))
                    removed_size += size
                    removed_count += 1
            except Exception as e:
                logger.warning("Error processing %s: %s", run_dir, e)

    action = "Would remove" if dry_run else "Removed"
    logger.info("%s %d runs older than %d days, total %s", action, removed_count, keep_days, format_size(removed_size))
    if dry_run:
        logger.info("Run with --execute to actually remove runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Storage cleanup utilities for ML experiments")
    parser.add_argument("--analyze", action="store_true", help="Analyze storage usage")
    parser.add_argument("--clean-predictions", action="store_true", help="Remove training prediction CSVs")
    parser.add_argument("--compress-models", action="store_true", help="Compress existing model files")
    parser.add_argument("--clean-mlruns", action="store_true", help="Clean old MLflow runs")
    parser.add_argument("--keep-days", type=int, default=30, help="Days to keep MLflow runs (default: 30)")
    parser.add_argument("--execute", action="store_true", help="Actually perform operations (default: dry run)")
    parser.add_argument("--base-path", type=str, default=".", help="Base path of the project")

    args = parser.parse_args()
    base_path = Path(args.base_path).resolve()
    dry_run = not args.execute

    if args.analyze or not any([args.clean_predictions, args.compress_models, args.clean_mlruns]):
        analysis = analyze_storage(base_path)
        print_analysis(analysis)

    if args.clean_predictions:
        clean_predictions(base_path, dry_run)

    if args.compress_models:
        compress_models(base_path, dry_run)

    if args.clean_mlruns:
        clean_mlruns(base_path, args.keep_days, dry_run)


if __name__ == "__main__":
    main()
