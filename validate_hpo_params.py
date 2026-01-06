#!/usr/bin/env python
"""Validate that Chemprop HPO parameters are correctly handled.

This script verifies that:
1. warmup_epochs is properly extracted from the config
2. lr_warmup_ratio and lr_final_ratio are handled correctly
3. aggregation parameter is removed from search space
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import after path modification
from admet.model.chemprop.hpo_config import ParameterSpace, SearchSpaceConfig  # noqa: E402
from admet.model.chemprop.hpo_search_space import build_search_space  # noqa: E402
from admet.model.chemprop.hpo_trainable import _build_hyperparams  # noqa: E402
from admet.model.chemprop.model import ChempropHyperparams  # noqa: E402


def test_search_space_no_aggregation():
    """Verify aggregation is not in search space."""
    config = SearchSpaceConfig(
        learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
        lr_warmup_ratio=ParameterSpace(type="uniform", low=0.1, high=1.0),
        lr_final_ratio=ParameterSpace(type="uniform", low=0.01, high=0.5),
        warmup_epochs=ParameterSpace(type="choice", values=[2, 3, 5]),
        batch_size=ParameterSpace(type="choice", values=[32, 64]),
    )
    space = build_search_space(config)

    # Verify warmup params are included
    assert "lr_warmup_ratio" in space, "lr_warmup_ratio should be in search space"
    assert "lr_final_ratio" in space, "lr_final_ratio should be in search space"
    assert "warmup_epochs" in space, "warmup_epochs should be in search space"

    # Verify aggregation is NOT included
    assert "aggregation" not in space, "aggregation should NOT be in search space"
    assert "aggregation_norm" not in space, "aggregation_norm should NOT be in search space"

    print("✅ Search space validation passed: aggregation removed, warmup params present")


def test_trainable_warmup_extraction():
    """Verify _build_hyperparams extracts warmup_epochs correctly."""
    config = {
        "learning_rate": 0.001,
        "lr_warmup_ratio": 0.1,
        "lr_final_ratio": 0.01,
        "warmup_epochs": 5,
        "batch_size": 64,
        "dropout": 0.1,
    }

    hyperparams = _build_hyperparams(config, max_epochs=100, seed=42)

    # Verify learning rate schedule
    assert hyperparams.max_lr == 0.001, f"Expected max_lr=0.001, got {hyperparams.max_lr}"
    assert hyperparams.init_lr == 0.0001, f"Expected init_lr=0.0001, got {hyperparams.init_lr}"
    assert hyperparams.final_lr == 0.00001, f"Expected final_lr=0.00001, got {hyperparams.final_lr}"

    # Verify warmup_epochs is extracted
    assert hyperparams.warmup_epochs == 5, f"Expected warmup_epochs=5, got {hyperparams.warmup_epochs}"

    # Verify other params
    assert hyperparams.batch_size == 64, f"Expected batch_size=64, got {hyperparams.batch_size}"
    assert hyperparams.dropout == 0.1, f"Expected dropout=0.1, got {hyperparams.dropout}"

    print("✅ Trainable extraction validation passed: warmup_epochs and LR schedule correct")


def test_chemprop_hyperparams_fields():
    """Verify ChempropHyperparams has all necessary fields."""
    params = ChempropHyperparams()

    # Verify warmup-related fields exist
    assert hasattr(params, "init_lr"), "ChempropHyperparams should have init_lr field"
    assert hasattr(params, "max_lr"), "ChempropHyperparams should have max_lr field"
    assert hasattr(params, "final_lr"), "ChempropHyperparams should have final_lr field"
    assert hasattr(params, "warmup_epochs"), "ChempropHyperparams should have warmup_epochs field"

    # Verify defaults
    assert params.warmup_epochs == 5, f"Expected default warmup_epochs=5, got {params.warmup_epochs}"

    print("✅ ChempropHyperparams validation passed: all warmup fields present with correct defaults")


if __name__ == "__main__":
    print("Validating Chemprop HPO parameter handling...\n")

    try:
        test_search_space_no_aggregation()
        test_trainable_warmup_extraction()
        test_chemprop_hyperparams_fields()

        print("\n✅ All validations passed!")
        print("\nSummary:")
        print("  - warmup_epochs is properly extracted and passed to ChempropHyperparams")
        print("  - lr_warmup_ratio and lr_final_ratio correctly compute init_lr and final_lr")
        print("  - aggregation parameter removed from search space (hardcoded to norm)")

    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
