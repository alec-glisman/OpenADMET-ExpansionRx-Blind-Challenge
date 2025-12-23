"""Tests for deterministic/reproducible model and ensemble outputs.

These tests verify that models produce identical outputs when trained and
predicted with the same seed and configuration. This is critical for:
- Reproducible research results
- Debugging and development
- Production model validation

The tests cover:
1. Classical models (XGBoost, LightGBM, CatBoost)
2. Chemprop neural network models
3. Ensemble predictions
4. Curriculum and joint samplers
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from admet.model.registry import ModelRegistry

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_and_register():
    """Clear registry and re-register all models."""
    ModelRegistry.clear()

    import admet.model.classical.catboost_model
    import admet.model.classical.lightgbm_model
    import admet.model.classical.xgboost_model

    importlib.reload(admet.model.classical.xgboost_model)
    importlib.reload(admet.model.classical.lightgbm_model)
    importlib.reload(admet.model.classical.catboost_model)

    yield
    ModelRegistry.clear()


@pytest.fixture
def sample_smiles() -> list[str]:
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CC(C)O",  # isopropanol
        "CCCC",  # butane
        "c1ccc(O)cc1",  # phenol
        "CC(=O)OC",  # methyl acetate
        "CCOCC",  # diethyl ether
        "CCN",  # ethylamine
        "CCCO",  # propanol
    ]


@pytest.fixture
def sample_targets() -> np.ndarray:
    """Sample target values (2 targets)."""
    np.random.seed(42)
    return np.random.randn(10, 2)


@pytest.fixture
def test_smiles() -> list[str]:
    """Test SMILES for prediction."""
    return ["CCO", "CCCO", "c1ccccc1"]


# ============================================================================
# Classical Model Determinism Tests
# ============================================================================


class TestXGBoostDeterminism:
    """Test XGBoost model reproducibility."""

    @pytest.fixture
    def xgboost_config(self):
        """XGBoost configuration with fixed seed."""
        return OmegaConf.create(
            {
                "model": {
                    "type": "xgboost",
                    "xgboost": {
                        "n_estimators": 10,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                        "random_state": 42,
                    },
                    "fingerprint": {"type": "morgan"},
                },
                "mlflow": {"enabled": False},
            }
        )

    def test_same_seed_same_predictions(self, xgboost_config, sample_smiles, sample_targets, test_smiles):
        """Two models with same seed should produce identical predictions."""
        from admet.model.classical import XGBoostModel

        # Train first model
        model1 = XGBoostModel(xgboost_config)
        model1.fit(sample_smiles, sample_targets)
        pred1 = model1.predict(test_smiles)

        # Train second model with same config
        model2 = XGBoostModel(xgboost_config)
        model2.fit(sample_smiles, sample_targets)
        pred2 = model2.predict(test_smiles)

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10, err_msg="XGBoost predictions should be identical with same seed"
        )

    def test_different_seed_different_predictions(self, sample_smiles, sample_targets, test_smiles):
        """Two models with different seeds should produce different predictions."""
        from admet.model.classical import XGBoostModel

        config1 = OmegaConf.create(
            {
                "model": {
                    "type": "xgboost",
                    "xgboost": {"n_estimators": 10, "random_state": 42},
                    "fingerprint": {"type": "morgan"},
                },
                "mlflow": {"enabled": False},
            }
        )
        config2 = OmegaConf.create(
            {
                "model": {
                    "type": "xgboost",
                    "xgboost": {"n_estimators": 10, "random_state": 123},
                    "fingerprint": {"type": "morgan"},
                },
                "mlflow": {"enabled": False},
            }
        )

        model1 = XGBoostModel(config1)
        model1.fit(sample_smiles, sample_targets)
        pred1 = model1.predict(test_smiles)

        model2 = XGBoostModel(config2)
        model2.fit(sample_smiles, sample_targets)
        pred2 = model2.predict(test_smiles)

        # Predictions should differ (not identical)
        assert not np.allclose(pred1, pred2), "Different seeds should produce different predictions"


class TestLightGBMDeterminism:
    """Test LightGBM model reproducibility."""

    @pytest.fixture
    def lightgbm_config(self):
        """LightGBM configuration with fixed seed."""
        return OmegaConf.create(
            {
                "model": {
                    "type": "lightgbm",
                    "lightgbm": {
                        "n_estimators": 10,
                        "num_leaves": 15,
                        "learning_rate": 0.1,
                        "random_state": 42,
                        "verbose": -1,
                    },
                    "fingerprint": {"type": "morgan"},
                },
                "mlflow": {"enabled": False},
            }
        )

    def test_same_seed_same_predictions(self, lightgbm_config, sample_smiles, sample_targets, test_smiles):
        """Two models with same seed should produce identical predictions."""
        from admet.model.classical import LightGBMModel

        model1 = LightGBMModel(lightgbm_config)
        model1.fit(sample_smiles, sample_targets)
        pred1 = model1.predict(test_smiles)

        model2 = LightGBMModel(lightgbm_config)
        model2.fit(sample_smiles, sample_targets)
        pred2 = model2.predict(test_smiles)

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10, err_msg="LightGBM predictions should be identical with same seed"
        )


class TestCatBoostDeterminism:
    """Test CatBoost model reproducibility."""

    @pytest.fixture
    def catboost_config(self):
        """CatBoost configuration with fixed seed."""
        return OmegaConf.create(
            {
                "model": {
                    "type": "catboost",
                    "catboost": {
                        "iterations": 10,
                        "depth": 3,
                        "learning_rate": 0.1,
                        "random_seed": 42,
                        "verbose": False,
                    },
                    "fingerprint": {"type": "morgan"},
                },
                "mlflow": {"enabled": False},
            }
        )

    def test_same_seed_same_predictions(self, catboost_config, sample_smiles, sample_targets, test_smiles):
        """Two models with same seed should produce identical predictions."""
        from admet.model.classical import CatBoostModel

        model1 = CatBoostModel(catboost_config)
        model1.fit(sample_smiles, sample_targets)
        pred1 = model1.predict(test_smiles)

        model2 = CatBoostModel(catboost_config)
        model2.fit(sample_smiles, sample_targets)
        pred2 = model2.predict(test_smiles)

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10, err_msg="CatBoost predictions should be identical with same seed"
        )


# ============================================================================
# Ensemble Determinism Tests
# ============================================================================


class TestEnsembleDeterminism:
    """Test ensemble prediction reproducibility."""

    @pytest.fixture
    def ensemble_config(self):
        """Ensemble configuration."""
        return OmegaConf.create(
            {
                "model": {
                    "type": "xgboost",
                    "xgboost": {"n_estimators": 5, "max_depth": 2, "random_state": 42},
                    "fingerprint": {"type": "morgan"},
                },
                "ensemble": {"n_models": 3, "aggregation": "mean"},
                "mlflow": {"enabled": False},
            }
        )

    def test_same_seeds_same_predictions(self, ensemble_config, sample_smiles, sample_targets, test_smiles):
        """Ensembles with same seeds should produce identical predictions."""
        from admet.model.ensemble import Ensemble

        seeds = [42, 123, 456]

        ensemble1 = Ensemble(ensemble_config)
        ensemble1.fit(sample_smiles, sample_targets, seeds=seeds)
        pred1 = ensemble1.predict(test_smiles)

        ensemble2 = Ensemble(ensemble_config)
        ensemble2.fit(sample_smiles, sample_targets, seeds=seeds)
        pred2 = ensemble2.predict(test_smiles)

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10, err_msg="Ensemble predictions should be identical with same seeds"
        )

    def test_same_seeds_same_uncertainty(self, ensemble_config, sample_smiles, sample_targets, test_smiles):
        """Ensembles with same seeds should produce identical uncertainty estimates."""
        from admet.model.ensemble import Ensemble

        seeds = [42, 123, 456]

        ensemble1 = Ensemble(ensemble_config)
        ensemble1.fit(sample_smiles, sample_targets, seeds=seeds)
        mean1, std1 = ensemble1.predict_with_uncertainty(test_smiles)

        ensemble2 = Ensemble(ensemble_config)
        ensemble2.fit(sample_smiles, sample_targets, seeds=seeds)
        mean2, std2 = ensemble2.predict_with_uncertainty(test_smiles)

        np.testing.assert_array_almost_equal(
            mean1, mean2, decimal=10, err_msg="Ensemble mean predictions should be identical"
        )
        np.testing.assert_array_almost_equal(std1, std2, decimal=10, err_msg="Ensemble uncertainty should be identical")

    def test_default_seeds_deterministic(self, ensemble_config, sample_smiles, sample_targets, test_smiles):
        """Ensembles with default seeds (0, 1, 2, ...) should be deterministic."""
        from admet.model.ensemble import Ensemble

        ensemble1 = Ensemble(ensemble_config)
        ensemble1.fit(sample_smiles, sample_targets)  # Uses default seeds [0, 1, 2]
        pred1 = ensemble1.predict(test_smiles)

        ensemble2 = Ensemble(ensemble_config)
        ensemble2.fit(sample_smiles, sample_targets)  # Uses default seeds [0, 1, 2]
        pred2 = ensemble2.predict(test_smiles)

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10, err_msg="Ensemble with default seeds should be deterministic"
        )

    def test_save_load_preserves_predictions(self, ensemble_config, sample_smiles, sample_targets, test_smiles):
        """Saved and loaded ensemble should produce identical predictions."""
        from admet.model.ensemble import Ensemble

        ensemble = Ensemble(ensemble_config)
        ensemble.fit(sample_smiles, sample_targets, seeds=[42, 123, 456])
        original_pred = ensemble.predict(test_smiles)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ensemble"
            ensemble.save(save_path)

            loaded_ensemble = Ensemble(ensemble_config)
            loaded_ensemble.load(save_path)
            loaded_pred = loaded_ensemble.predict(test_smiles)

        np.testing.assert_array_almost_equal(
            original_pred, loaded_pred, decimal=10, err_msg="Loaded ensemble should match original"
        )


# ============================================================================
# Sampler Determinism Tests
# ============================================================================


class TestSamplerDeterminism:
    """Test sampler reproducibility."""

    @pytest.fixture
    def quality_labels(self) -> list[str]:
        """Sample quality labels."""
        return ["high", "high", "medium", "low", "high", "medium", "low", "high", "low", "medium"]

    @pytest.fixture
    def curriculum_state(self):
        """Curriculum state for testing."""
        from admet.model.chemprop.curriculum import CurriculumState

        return CurriculumState(qualities=["high", "medium", "low"])

    def test_curriculum_sampler_deterministic(self, quality_labels, curriculum_state):
        """Curriculum sampler with same seed should produce identical indices."""
        from admet.model.chemprop.curriculum_sampler import build_curriculum_sampler

        sampler1 = build_curriculum_sampler(quality_labels, curriculum_state, num_samples=20, seed=42)
        indices1 = list(sampler1)

        sampler2 = build_curriculum_sampler(quality_labels, curriculum_state, num_samples=20, seed=42)
        indices2 = list(sampler2)

        assert indices1 == indices2, "Curriculum sampler should be deterministic with same seed"

    def test_curriculum_sampler_different_seeds(self, quality_labels, curriculum_state):
        """Curriculum sampler with different seeds should produce different indices."""
        from admet.model.chemprop.curriculum_sampler import build_curriculum_sampler

        sampler1 = build_curriculum_sampler(quality_labels, curriculum_state, num_samples=20, seed=42)
        indices1 = list(sampler1)

        sampler2 = build_curriculum_sampler(quality_labels, curriculum_state, num_samples=20, seed=123)
        indices2 = list(sampler2)

        assert indices1 != indices2, "Different seeds should produce different sampling"

    def test_dynamic_curriculum_sampler_deterministic(self, quality_labels, curriculum_state):
        """Dynamic curriculum sampler with same seed should be deterministic."""
        from admet.model.chemprop.curriculum_sampler import DynamicCurriculumSampler

        sampler1 = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=20,
            seed=42,
            increment_seed_per_epoch=False,
        )
        indices1 = list(iter(sampler1))

        sampler2 = DynamicCurriculumSampler(
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            num_samples=20,
            seed=42,
            increment_seed_per_epoch=False,
        )
        indices2 = list(iter(sampler2))

        assert indices1 == indices2, "Dynamic sampler should be deterministic with same seed"

    def test_joint_sampler_deterministic(self, quality_labels, curriculum_state):
        """Joint sampler with same seed should be deterministic."""
        from admet.model.chemprop.joint_sampler import JointSampler

        np.random.seed(42)
        targets = np.random.randn(10, 2)

        sampler1 = JointSampler(
            targets=targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.3,
            num_samples=20,
            seed=42,
            increment_seed_per_epoch=False,
        )
        indices1 = list(iter(sampler1))

        sampler2 = JointSampler(
            targets=targets,
            quality_labels=quality_labels,
            curriculum_state=curriculum_state,
            task_alpha=0.3,
            num_samples=20,
            seed=42,
            increment_seed_per_epoch=False,
        )
        indices2 = list(iter(sampler2))

        assert indices1 == indices2, "Joint sampler should be deterministic with same seed"


# ============================================================================
# Chemprop Model Determinism Tests (Requires GPU or longer runtime)
# ============================================================================


@pytest.mark.skipif(os.environ.get("RUN_E2E") != "1", reason="E2E tests skipped by default (set RUN_E2E=1)")
class TestChempropDeterminism:
    """Test Chemprop model reproducibility.

    These tests require more resources and are skipped by default.
    Run with RUN_E2E=1 pytest tests/test_determinism.py::TestChempropDeterminism
    """

    @pytest.fixture
    def sample_df(self, sample_smiles):
        """Sample DataFrame for Chemprop."""
        import pandas as pd

        np.random.seed(42)
        return pd.DataFrame(
            {
                "SMILES": sample_smiles,
                "LogD": np.random.randn(len(sample_smiles)),
            }
        )

    def test_same_seed_same_predictions(self, sample_df):
        """Chemprop with same seed should produce identical predictions."""
        from admet.model.chemprop.model import ChempropHyperparams, ChempropModel

        train_df = sample_df.iloc[:-3].copy()
        val_df = sample_df.iloc[-3:].copy()
        test_smiles = ["CCO", "CCCO"]

        hyperparams = ChempropHyperparams(
            max_epochs=2,
            batch_size=4,
            num_workers=0,
            seed=42,
        )

        # Train first model
        model1 = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            smiles_col="SMILES",
            target_cols=["LogD"],
            target_weights=[1.0],
            progress_bar=False,
            hyperparams=hyperparams,
            mlflow_tracking=False,
        )
        model1.fit()
        pred1 = model1.predict_smiles(test_smiles)

        # Train second model with same config
        model2 = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            smiles_col="SMILES",
            target_cols=["LogD"],
            target_weights=[1.0],
            progress_bar=False,
            hyperparams=hyperparams,
            mlflow_tracking=False,
        )
        model2.fit()
        pred2 = model2.predict_smiles(test_smiles)

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=5, err_msg="Chemprop predictions should be similar with same seed"
        )

    def test_save_load_preserves_predictions(self, sample_df):
        """Saved and loaded Chemprop model should produce identical predictions."""
        from admet.model.chemprop.model import ChempropHyperparams, ChempropModel

        train_df = sample_df.iloc[:-3].copy()
        val_df = sample_df.iloc[-3:].copy()
        test_smiles = ["CCO", "CCCO"]

        hyperparams = ChempropHyperparams(
            max_epochs=2,
            batch_size=4,
            num_workers=0,
            seed=42,
        )

        model = ChempropModel(
            df_train=train_df,
            df_validation=val_df,
            smiles_col="SMILES",
            target_cols=["LogD"],
            target_weights=[1.0],
            progress_bar=False,
            hyperparams=hyperparams,
            mlflow_tracking=False,
        )
        model.fit()
        original_pred = model.predict_smiles(test_smiles)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "chemprop_model.pt"
            model.save(save_path)

            # Load into new model instance
            loaded_model = ChempropModel(
                df_train=train_df,
                df_validation=val_df,
                smiles_col="SMILES",
                target_cols=["LogD"],
                target_weights=[1.0],
                progress_bar=False,
                hyperparams=hyperparams,
                mlflow_tracking=False,
            )
            loaded_model.load(save_path)
            loaded_pred = loaded_model.predict_smiles(test_smiles)

        np.testing.assert_array_almost_equal(
            original_pred, loaded_pred, decimal=10, err_msg="Loaded model should match original predictions"
        )


# ============================================================================
# Cross-run Reproducibility Tests
# ============================================================================


class TestCrossRunReproducibility:
    """Test that results are reproducible across separate runs."""

    def test_numpy_random_state_isolation(self):
        """Verify that numpy random state is properly isolated."""
        # This tests that setting a seed gives consistent results
        np.random.seed(42)
        arr1 = np.random.randn(10)

        np.random.seed(42)
        arr2 = np.random.randn(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_fingerprint_determinism(self, sample_smiles):
        """Verify fingerprint generation is deterministic."""
        import pandas as pd

        from admet.data.fingerprint import MorganFingerprintGenerator

        generator = MorganFingerprintGenerator(radius=2, fp_size=1024)

        smiles_series = pd.Series(sample_smiles)

        fp1 = generator.calculate_fingerprints(smiles_series)
        fp2 = generator.calculate_fingerprints(smiles_series)

        np.testing.assert_array_equal(fp1.values, fp2.values, err_msg="Fingerprints should be deterministic")

    def test_model_registry_determinism(self, sample_smiles, sample_targets):
        """Verify ModelRegistry creates deterministic models."""
        config = OmegaConf.create(
            {
                "model": {
                    "type": "xgboost",
                    "xgboost": {"n_estimators": 5, "random_state": 42},
                    "fingerprint": {"type": "morgan"},
                },
                "mlflow": {"enabled": False},
            }
        )

        model1 = ModelRegistry.create(config)
        model1.fit(sample_smiles, sample_targets)
        pred1 = model1.predict(sample_smiles[:3])

        model2 = ModelRegistry.create(config)
        model2.fit(sample_smiles, sample_targets)
        pred2 = model2.predict(sample_smiles[:3])

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10, err_msg="ModelRegistry should create deterministic models"
        )
