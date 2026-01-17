#!/usr/bin/env python3
"""Debug script to verify PerQualityMetricsCallback is working correctly.

This script helps diagnose why per-quality metrics may not be appearing in MLflow.
It performs several checks:
1. Verifies callback is instantiated
2. Checks validation dataloader is accessible
3. Verifies quality labels are provided
4. Tests the callback in isolation
5. Checks MLflow configuration

Usage:
    python scripts/analysis/debug_per_quality_metrics.py

Requirements:
    - Training configuration with log_per_quality_metrics: true
    - MLflow server running (optional, for full test)
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch

# Setup logging to see detailed output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_callback_import():
    """Test 1: Can we import the callback?"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Import PerQualityMetricsCallback")
    logger.info("=" * 80)
    try:
        from admet.model.chemprop.curriculum import PerQualityMetricsCallback  # noqa: F401

        logger.info("✓ Successfully imported PerQualityMetricsCallback")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import: {e}")
        return False


def test_callback_instantiation():
    """Test 2: Can we create the callback with quality labels?"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Instantiate callback with quality labels")
    logger.info("=" * 80)
    try:
        from admet.model.chemprop.curriculum import PerQualityMetricsCallback

        quality_labels = ["high"] * 50 + ["medium"] * 30 + ["low"] * 20
        qualities = ["high", "medium", "low"]
        target_cols = ["LogD", "KSOL"]

        callback = PerQualityMetricsCallback(
            val_quality_labels=quality_labels, qualities=qualities, target_cols=target_cols, compute_every_n_epochs=1
        )

        logger.info(f"✓ Callback created with {len(quality_labels)} quality labels")
        logger.info(f"  Qualities: {qualities}")
        logger.info(f"  Target columns: {target_cols}")
        logger.info(
            f"  Quality distribution: high={quality_labels.count('high')}, "
            f"medium={quality_labels.count('medium')}, low={quality_labels.count('low')}"
        )
        return True, callback
    except Exception as e:
        logger.error(f"✗ Failed to instantiate: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False, None


def test_callback_with_mock_data():
    """Test 3: Run callback with mock PyTorch Lightning trainer and module."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Run callback with mock data")
    logger.info("=" * 80)
    try:
        from admet.model.chemprop.curriculum import PerQualityMetricsCallback

        # Create callback
        quality_labels = ["high"] * 50 + ["medium"] * 30 + ["low"] * 20
        qualities = ["high", "medium", "low"]
        target_cols = ["LogD", "KSOL"]
        callback = PerQualityMetricsCallback(
            val_quality_labels=quality_labels, qualities=qualities, target_cols=target_cols, compute_every_n_epochs=1
        )

        # Create mock module
        mock_preds = torch.tensor([[1.0, 2.0]] * 100)  # 100 samples, 2 targets
        mock_module = MagicMock()
        mock_module.current_epoch = 0
        mock_module.return_value = mock_preds

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_module.parameters.return_value = iter([mock_param])
        mock_module.eval = MagicMock()

        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0

        # Create mock batch (100 samples, 2 targets)
        batch_targets = torch.randn(100, 2)
        mock_batch = (
            MagicMock(),  # bmg
            None,  # V_d
            None,  # X_d
            batch_targets,
            torch.ones(100, 2),  # weights
            None,  # lt_mask
            None,  # gt_mask
        )
        mock_batch[0].to = MagicMock(return_value=mock_batch[0])

        # Mock dataloader
        class MockDataLoader:
            def __iter__(self):
                yield mock_batch

        mock_trainer.val_dataloaders = MockDataLoader()

        # Run callback
        logger.info("Calling callback.on_validation_epoch_end()...")
        callback.on_validation_epoch_end(mock_trainer, mock_module)

        # Check if metrics were logged
        if mock_module.log.called:
            logged_metrics = [call[0][0] for call in mock_module.log.call_args_list]
            logger.info(f"✓ Callback logged {len(logged_metrics)} metrics")
            logger.info(f"  Sample metrics: {logged_metrics[:5]}")

            # Check for expected metric hierarchy
            expected_patterns = ["val/mae/high", "val/mae/medium", "val/rmse/low"]
            found = [p for p in expected_patterns if p in logged_metrics]
            if found:
                logger.info(f"✓ Found expected metric patterns: {found}")
            else:
                logger.warning(f"⚠ Expected patterns not found: {expected_patterns}")
                logger.warning(f"  Actual metrics: {logged_metrics}")
        else:
            logger.warning("⚠ No metrics were logged to pl_module.log()")

        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def check_config_file():
    """Test 4: Check if config has log_per_quality_metrics enabled."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Check configuration file")
    logger.info("=" * 80)
    try:
        import yaml

        config_path = Path("configs/curriculum/chemprop_curriculum.yaml")

        if not config_path.exists():
            logger.warning(f"⚠ Config file not found: {config_path}")
            return False

        with open(config_path) as f:
            config = yaml.safe_load(f)

        log_metrics = config.get("joint_sampling", {}).get("curriculum", {}).get("log_per_quality_metrics", False)
        logger.info(f"log_per_quality_metrics: {log_metrics}")

        if log_metrics:
            logger.info("✓ Config has log_per_quality_metrics enabled")
        else:
            logger.warning("✗ Config has log_per_quality_metrics disabled or missing")

        # Check other relevant settings
        enabled = config.get("joint_sampling", {}).get("curriculum", {}).get("enabled", False)
        logger.info(f"curriculum.enabled: {enabled}")

        quality_col = config.get("joint_sampling", {}).get("curriculum", {}).get("quality_col")
        logger.info(f"curriculum.quality_col: {quality_col}")

        qualities = config.get("joint_sampling", {}).get("curriculum", {}).get("qualities", [])
        logger.info(f"curriculum.qualities: {qualities}")

        return log_metrics
    except Exception as e:
        logger.error(f"✗ Failed to check config: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def check_mlflow_setup():
    """Test 5: Check MLflow configuration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Check MLflow setup")
    logger.info("=" * 80)
    try:
        import mlflow

        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {tracking_uri}")

        # Try to check if server is accessible
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            logger.info(f"✓ MLflow server accessible, found {len(experiments)} experiments")
            return True
        except Exception as e:
            logger.warning(f"⚠ MLflow server may not be accessible: {e}")
            return False

    except Exception as e:
        logger.error(f"✗ MLflow check failed: {e}")
        return False


def main():
    """Run all diagnostic tests."""
    logger.info("\n" + "=" * 80)
    logger.info("PerQualityMetricsCallback Diagnostic Tests")
    logger.info("=" * 80)

    results = []

    # Test 1: Import
    results.append(("Import", test_callback_import()))

    # Test 2: Instantiation
    success, callback = test_callback_instantiation()
    results.append(("Instantiation", success))

    # Test 3: Mock data
    results.append(("Mock Data", test_callback_with_mock_data()))

    # Test 4: Config
    results.append(("Config", check_config_file()))

    # Test 5: MLflow
    results.append(("MLflow", check_mlflow_setup()))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{name:20s}: {status}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        logger.info("\n✓ All tests passed! Callback should work in training.")
        logger.info("\nNext steps:")
        logger.info("1. Run your training with logging level INFO to see callback output")
        logger.info("2. Check trainer.callbacks list to verify callback is added")
        logger.info("3. Verify validation is actually running (check logs)")
        return 0
    else:
        logger.warning("\n⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
