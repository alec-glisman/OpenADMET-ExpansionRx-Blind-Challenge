"""Tests for generic Ensemble class."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from admet.model.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clear_and_register():
    """Clear registry and re-register models."""
    ModelRegistry.clear()

    # Import to trigger registration (no reload needed after clear)
    import admet.model.classical.xgboost_model  # noqa: F401

    # Only reload if not registered
    if not ModelRegistry.is_registered("xgboost"):
        importlib.reload(admet.model.classical.xgboost_model)

    yield
    ModelRegistry.clear()


from admet.model.ensemble import Ensemble  # noqa: E402


@pytest.fixture
def sample_smiles():
    """Sample SMILES for testing."""
    return [
        "CCO",
        "CCCO",
        "CCCCO",
        "c1ccccc1",
        "CC(=O)O",
        "CC(C)O",
        "CCN",
        "CCC",
    ]


@pytest.fixture
def sample_targets():
    """Sample target values for testing."""
    return np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0], [3.5, 4.5], [4.0, 5.0], [4.5, 5.5]])


@pytest.fixture
def ensemble_config():
    """Ensemble configuration."""
    return OmegaConf.create(
        {
            "model": {
                "type": "xgboost",
                "xgboost": {"n_estimators": 5, "max_depth": 2},
                "fingerprint": {"type": "morgan"},
            },
            "ensemble": {"n_models": 3, "aggregation": "mean"},
            "mlflow": {"enabled": False},
        }
    )


class TestEnsemble:
    """Tests for Ensemble class."""

    def test_init(self, ensemble_config):
        """Test ensemble initialization."""
        ensemble = Ensemble(ensemble_config)

        assert ensemble.model_type == "xgboost"
        assert ensemble.n_models == 3
        assert ensemble.aggregation == "mean"
        assert not ensemble.is_fitted

    def test_fit(self, ensemble_config, sample_smiles, sample_targets):
        """Test ensemble fitting."""
        ensemble = Ensemble(ensemble_config)

        ensemble.fit(sample_smiles, sample_targets)

        assert ensemble.is_fitted
        assert len(ensemble.models) == 3
        assert all(m.is_fitted for m in ensemble.models)

    def test_predict(self, ensemble_config, sample_smiles, sample_targets):
        """Test ensemble prediction."""
        ensemble = Ensemble(ensemble_config)
        ensemble.fit(sample_smiles, sample_targets)

        predictions = ensemble.predict(sample_smiles)

        assert predictions.shape == sample_targets.shape

    def test_predict_without_fit_raises(self, ensemble_config):
        """Test predict raises if not fitted."""
        ensemble = Ensemble(ensemble_config)

        with pytest.raises(RuntimeError, match="not been fitted"):
            ensemble.predict(["CCO"])

    def test_predict_with_uncertainty(self, ensemble_config, sample_smiles, sample_targets):
        """Test prediction with uncertainty."""
        ensemble = Ensemble(ensemble_config)
        ensemble.fit(sample_smiles, sample_targets)

        mean_pred, std_pred = ensemble.predict_with_uncertainty(sample_smiles)

        assert mean_pred.shape == sample_targets.shape
        assert std_pred.shape == sample_targets.shape
        assert np.all(std_pred >= 0)

    def test_median_aggregation(self, sample_smiles, sample_targets):
        """Test median aggregation."""
        config = OmegaConf.create(
            {
                "model": {
                    "type": "xgboost",
                    "xgboost": {"n_estimators": 5},
                    "fingerprint": {"type": "morgan"},
                },
                "ensemble": {"n_models": 3, "aggregation": "median"},
                "mlflow": {"enabled": False},
            }
        )

        ensemble = Ensemble(config)
        ensemble.fit(sample_smiles, sample_targets)

        predictions = ensemble.predict(sample_smiles)

        assert predictions.shape == sample_targets.shape

    def test_save_load(self, ensemble_config, sample_smiles, sample_targets):
        """Test saving and loading ensemble."""
        ensemble = Ensemble(ensemble_config)
        ensemble.fit(sample_smiles, sample_targets)

        original_pred = ensemble.predict(sample_smiles)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "ensemble"
            ensemble.save(save_path)

            loaded_ensemble = Ensemble(ensemble_config)
            loaded_ensemble.load(save_path)

            loaded_pred = loaded_ensemble.predict(sample_smiles)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_from_config(self, ensemble_config):
        """Test creating ensemble from config."""
        ensemble = Ensemble.from_config(ensemble_config)

        assert ensemble.model_type == "xgboost"
        assert ensemble.n_models == 3

    def test_custom_seeds(self, ensemble_config, sample_smiles, sample_targets):
        """Test ensemble with custom seeds."""
        ensemble = Ensemble(ensemble_config)

        custom_seeds = [42, 123, 456]
        ensemble.fit(sample_smiles, sample_targets, seeds=custom_seeds)

        assert ensemble.is_fitted
        assert len(ensemble.models) == 3
