"""Tests for base model classes: BaseModel, ModelRegistry, MLflowMixin."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from admet.model.base import BaseModel
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

if TYPE_CHECKING:
    from omegaconf import DictConfig


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_config() -> DictConfig:
    """Create a sample configuration for testing."""
    return OmegaConf.create(
        {
            "model": {
                "type": "test_model",
                "test_model": {"param1": 10, "param2": "value"},
            },
            "mlflow": {
                "enabled": True,
                "experiment_name": "test_experiment",
                "run_name": "test_run",
            },
        }
    )


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the model registry before and after each test."""
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


# ============================================================================
# Concrete Implementation for Testing
# ============================================================================


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    model_type = "concrete"

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: list[str] | None = None,
        val_y: np.ndarray | None = None,
    ) -> "ConcreteModel":
        """Simple fit implementation for testing."""
        self._fitted = True
        self._train_size = len(smiles)
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Simple predict implementation for testing."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return np.zeros(len(smiles))

    @classmethod
    def from_config(cls, config: DictConfig) -> "ConcreteModel":
        """Create model from config."""
        return cls(config)


class ConcreteModelWithMixin(BaseModel, MLflowMixin):
    """Concrete model with MLflow mixin for testing."""

    model_type = "concrete_mlflow"

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: list[str] | None = None,
        val_y: np.ndarray | None = None,
    ) -> "ConcreteModelWithMixin":
        """Fit with MLflow tracking."""
        self.init_mlflow()
        self.log_params_from_config()
        self.log_metrics({"train_size": len(smiles)})
        self._fitted = True
        self.end_mlflow()
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Predict implementation."""
        return np.zeros(len(smiles))

    @classmethod
    def from_config(cls, config: DictConfig) -> "ConcreteModelWithMixin":
        """Create from config."""
        return cls(config)


# ============================================================================
# BaseModel Tests
# ============================================================================


class TestBaseModel:
    """Tests for BaseModel abstract class."""

    def test_init(self, sample_config: DictConfig):
        """Test BaseModel initialization."""
        model = ConcreteModel(sample_config)
        assert model.config == sample_config
        assert model._fitted is False
        assert model.is_fitted is False

    def test_fit_sets_fitted(self, sample_config: DictConfig):
        """Test that fit() sets the fitted flag."""
        model = ConcreteModel(sample_config)
        smiles = ["CCO", "CCCO"]
        y = np.array([1.0, 2.0])

        model.fit(smiles, y)

        assert model._fitted is True
        assert model.is_fitted is True

    def test_predict_requires_fit(self, sample_config: DictConfig):
        """Test that predict() raises if model not fitted."""
        model = ConcreteModel(sample_config)
        smiles = ["CCO", "CCCO"]

        with pytest.raises(RuntimeError, match="Model not fitted"):
            model.predict(smiles)

    def test_predict_after_fit(self, sample_config: DictConfig):
        """Test predict() works after fitting."""
        model = ConcreteModel(sample_config)
        smiles = ["CCO", "CCCO"]
        y = np.array([1.0, 2.0])

        model.fit(smiles, y)
        predictions = model.predict(smiles)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(smiles)

    def test_get_params(self, sample_config: DictConfig):
        """Test sklearn-compatible get_params()."""
        model = ConcreteModel(sample_config)
        params = model.get_params()

        assert "config" in params
        assert params["config"] == sample_config

    def test_set_params(self, sample_config: DictConfig):
        """Test sklearn-compatible set_params()."""
        model = ConcreteModel(sample_config)
        model.set_params(custom_attr="test_value")

        assert hasattr(model, "custom_attr")
        assert model.custom_attr == "test_value"

    def test_repr(self, sample_config: DictConfig):
        """Test string representation."""
        model = ConcreteModel(sample_config)
        repr_str = repr(model)

        assert "ConcreteModel" in repr_str
        assert "fitted=False" in repr_str


# ============================================================================
# ModelRegistry Tests
# ============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_decorator(self, sample_config: DictConfig):
        """Test @ModelRegistry.register() decorator."""

        @ModelRegistry.register("my_model")
        class MyModel(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        assert "my_model" in ModelRegistry.list_models()
        assert ModelRegistry.get("my_model") == MyModel
        assert MyModel.model_type == "my_model"

    def test_register_duplicate_raises(self, sample_config: DictConfig):
        """Test that registering same type twice raises ValueError."""

        @ModelRegistry.register("duplicate")
        class Model1(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        with pytest.raises(ValueError, match="already registered"):

            @ModelRegistry.register("duplicate")
            class Model2(BaseModel):
                def fit(self, smiles, y, val_smiles=None, val_y=None):
                    return self

                def predict(self, smiles):
                    return np.zeros(len(smiles))

                @classmethod
                def from_config(cls, config):
                    return cls(config)

    def test_create_model(self, sample_config: DictConfig):
        """Test ModelRegistry.create() factory method."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        model = ModelRegistry.create(sample_config)

        assert isinstance(model, TestModel)
        assert model.config == sample_config

    def test_create_unknown_type_raises(self, sample_config: DictConfig):
        """Test that creating unknown type raises ValueError."""
        config = OmegaConf.create({"model": {"type": "unknown_model"}})

        with pytest.raises(ValueError, match="Unknown model type"):
            ModelRegistry.create(config)

    def test_list_models(self):
        """Test listing registered models."""

        @ModelRegistry.register("model_a")
        class ModelA(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        @ModelRegistry.register("model_b")
        class ModelB(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        models = ModelRegistry.list_models()

        assert "model_a" in models
        assert "model_b" in models
        assert len(models) == 2

    def test_get_nonexistent_raises(self):
        """Test that getting non-existent model raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            ModelRegistry.get("nonexistent")

    def test_is_registered(self):
        """Test is_registered() method."""
        assert ModelRegistry.is_registered("not_registered") is False

        @ModelRegistry.register("registered")
        class RegisteredModel(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        assert ModelRegistry.is_registered("registered") is True

    def test_clear(self):
        """Test clearing the registry."""

        @ModelRegistry.register("to_clear")
        class ToClearModel(BaseModel):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                return self

            def predict(self, smiles):
                return np.zeros(len(smiles))

            @classmethod
            def from_config(cls, config):
                return cls(config)

        assert len(ModelRegistry.list_models()) > 0

        ModelRegistry.clear()

        assert len(ModelRegistry.list_models()) == 0


# ============================================================================
# MLflowMixin Tests
# ============================================================================


class TestMLflowMixin:
    """Tests for MLflowMixin."""

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_init_mlflow(self, mock_mlflow, sample_config: DictConfig):
        """Test MLflow initialization."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        model = ConcreteModelWithMixin(sample_config)
        run_id = model.init_mlflow(run_name="test")

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once()
        assert run_id == "test_run_id"

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_init_mlflow_disabled(self, mock_mlflow):
        """Test MLflow initialization when disabled."""
        config = OmegaConf.create(
            {
                "model": {"type": "test"},
                "mlflow": {"enabled": False},
            }
        )

        model = ConcreteModelWithMixin(config)
        run_id = model.init_mlflow()

        mock_mlflow.start_run.assert_not_called()
        assert run_id is None

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_log_params_from_config(self, mock_mlflow, sample_config: DictConfig):
        """Test logging config parameters."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        model = ConcreteModelWithMixin(sample_config)
        model.init_mlflow()
        model.log_params_from_config()

        mock_mlflow.log_params.assert_called()

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_log_metrics(self, mock_mlflow, sample_config: DictConfig):
        """Test logging metrics."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        model = ConcreteModelWithMixin(sample_config)
        model.init_mlflow()
        model.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=10)

        mock_mlflow.log_metrics.assert_called_with({"loss": 0.5, "accuracy": 0.9}, step=10)

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_log_metric(self, mock_mlflow, sample_config: DictConfig):
        """Test logging single metric."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        model = ConcreteModelWithMixin(sample_config)
        model.init_mlflow()
        model.log_metric("test_metric", 0.42, step=5)

        mock_mlflow.log_metric.assert_called_with("test_metric", 0.42, step=5)

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_end_mlflow(self, mock_mlflow, sample_config: DictConfig):
        """Test ending MLflow run."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        model = ConcreteModelWithMixin(sample_config)
        model.init_mlflow()
        model.end_mlflow()

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
        assert model._mlflow_run_id is None

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_no_logging_without_init(self, mock_mlflow, sample_config: DictConfig):
        """Test that logging methods do nothing without init."""
        model = ConcreteModelWithMixin(sample_config)

        # These should not raise and should not call mlflow
        model.log_metrics({"test": 1.0})
        model.log_metric("test", 1.0)
        model.log_params({"param": "value"})

        mock_mlflow.log_metrics.assert_not_called()
        mock_mlflow.log_metric.assert_not_called()
        mock_mlflow.log_params.assert_not_called()

    def test_flatten_dict(self):
        """Test dictionary flattening utility."""
        nested = {
            "a": 1,
            "b": {"c": 2, "d": {"e": 3}},
        }

        flat = MLflowMixin._flatten_dict(nested)

        assert flat["a"] == 1
        assert flat["b.c"] == 2
        assert flat["b.d.e"] == 3

    def test_flatten_dict_max_depth(self):
        """Test max_depth parameter for flattening."""
        nested = {
            "a": {"b": {"c": {"d": 4}}},
        }

        flat = MLflowMixin._flatten_dict(nested, max_depth=2)

        assert "a.b.c" in flat
        # At max_depth=2, we should see the dict as value, not flattened further
        assert flat["a.b.c"] == {"d": 4}


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for base classes working together."""

    @patch("admet.model.mlflow_mixin.mlflow")
    def test_full_workflow(self, mock_mlflow, sample_config: DictConfig):
        """Test complete workflow: register, create, fit, predict."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value = mock_run

        # Register model
        @ModelRegistry.register("workflow_test")
        class WorkflowModel(BaseModel, MLflowMixin):
            def fit(self, smiles, y, val_smiles=None, val_y=None):
                self.init_mlflow()
                self.log_params_from_config()
                self._fitted = True
                self.log_metrics({"train_loss": 0.1})
                self.end_mlflow()
                return self

            def predict(self, smiles):
                return np.ones(len(smiles)) * 0.5

            @classmethod
            def from_config(cls, config):
                return cls(config)

        # Create config
        config = OmegaConf.create(
            {
                "model": {"type": "workflow_test"},
                "mlflow": {
                    "enabled": True,
                    "experiment_name": "integration_test",
                },
            }
        )

        # Create model from registry
        model = ModelRegistry.create(config)
        assert isinstance(model, WorkflowModel)

        # Fit model
        smiles = ["CCO", "CCCO", "CCCCO"]
        y = np.array([1.0, 2.0, 3.0])
        model.fit(smiles, y)

        assert model.is_fitted

        # Make predictions
        predictions = model.predict(smiles)
        assert len(predictions) == len(smiles)
        assert np.allclose(predictions, 0.5)
