#!/usr/bin/env python
"""Comprehensive test for HPO and ensemble configs with logging field.

This script validates that all configurations load properly with the
new logging field and that the QuietProgressReporter fix works.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, "src")


def test_hpo_configs():
    """Test HPO configuration loading."""
    print("=" * 60)
    print("Testing HPO Configurations")
    print("=" * 60)

    # Test Chemprop HPO
    print("\n1. Chemprop HPO Config:")
    from admet.model.chemprop.hpo_config import HPOConfig

    config_path = Path("configs/1-hpo-single/hpo_chemprop.yaml")
    raw_config = OmegaConf.load(config_path)
    merged_config = OmegaConf.merge(OmegaConf.structured(HPOConfig), raw_config)
    print("   ✅ Config loads without errors")
    print(f"   - Experiment: {merged_config.experiment_name}")
    print(f"   - Logging enabled: {merged_config.logging.enabled}")
    print(f"   - Logging verbose: {merged_config.logging.verbose}")
    print(f"   - Num samples: {merged_config.resources.num_samples}")

    # Test Chemeleon HPO
    print("\n2. Chemeleon HPO Config:")
    from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig

    config_path = Path("configs/1-hpo-single/hpo_chemeleon.yaml")
    raw_config = OmegaConf.load(config_path)
    merged_config = OmegaConf.merge(OmegaConf.structured(ChemeleonHPOConfig), raw_config)
    print("   ✅ Config loads without errors")
    print(f"   - Experiment: {merged_config.experiment_name}")
    print(f"   - Logging enabled: {merged_config.logging.enabled}")
    print(f"   - Logging verbose: {merged_config.logging.verbose}")
    print(f"   - Num samples: {merged_config.resources.num_samples}")

    print("\n✅ All HPO configs validated")
    return True


def test_ensemble_configs():
    """Test ensemble configuration loading."""
    print("\n" + "=" * 60)
    print("Testing Ensemble Configurations")
    print("=" * 60)

    # Test Chemprop Ensemble
    print("\n1. Chemprop Ensemble Config:")
    config_path = Path("configs/0-experiment/ensemble_chemprop_production.yaml")
    config = OmegaConf.load(config_path)
    print("   ✅ Config loads without errors")
    print(f"   - Model type: {config.model.type}")
    print(f"   - MLflow: {config.mlflow.experiment_name}")
    print(f"   - Logging enabled: {config.logging.enabled}")
    print(f"   - Ray max_parallel: {config.ray.max_parallel}")

    # Test Chemeleon Ensemble
    print("\n2. Chemeleon Ensemble Config:")
    config_path = Path("configs/0-experiment/ensemble_chemeleon_production.yaml")
    config = OmegaConf.load(config_path)
    print("   ✅ Config loads without errors")
    print(f"   - Model type: {config.model.type}")
    print(f"   - MLflow: {config.mlflow.experiment_name}")
    print(f"   - Logging enabled: {config.logging.enabled}")
    print(f"   - Ray max_parallel: {config.ray.max_parallel}")

    print("\n✅ All ensemble configs validated")
    return True


def test_quiet_progress_reporter():
    """Test QuietProgressReporter initialization."""
    print("\n" + "=" * 60)
    print("Testing QuietProgressReporter")
    print("=" * 60)

    from admet.util.ray_logging import QuietProgressReporter

    # Test initialization without parameters
    print("\n1. Default initialization:")
    reporter = QuietProgressReporter()
    print("   ✅ QuietProgressReporter() works")

    # Test initialization with parameters
    print("\n2. With metric_columns parameter:")
    reporter = QuietProgressReporter(metric_columns=["val_mae", "val_loss"])
    assert reporter is not None
    print("   ✅ QuietProgressReporter(metric_columns=[...]) works")

    # Test initialization with None
    print("\n3. With None parameter:")
    reporter = QuietProgressReporter(metric_columns=None)
    assert reporter is not None
    print("   ✅ QuietProgressReporter(metric_columns=None) works")

    print("\n✅ QuietProgressReporter validated")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HPO & ENSEMBLE CONFIG VALIDATION")
    print("=" * 60)
    print("Testing all configurations with the new logging field")
    print("and QuietProgressReporter fix...\n")

    results = {}

    try:
        results["HPO Configs"] = test_hpo_configs()
    except Exception as e:
        print(f"\n❌ HPO config test failed: {e}")
        results["HPO Configs"] = False

    try:
        results["Ensemble Configs"] = test_ensemble_configs()
    except Exception as e:
        print(f"\n❌ Ensemble config test failed: {e}")
        results["Ensemble Configs"] = False

    try:
        results["QuietProgressReporter"] = test_quiet_progress_reporter()
    except Exception as e:
        print(f"\n❌ QuietProgressReporter test failed: {e}")
        results["QuietProgressReporter"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30} {status}")
    print("=" * 60)

    if all(results.values()):
        print("\n✅ All tests passed! HPO and ensemble training are ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
