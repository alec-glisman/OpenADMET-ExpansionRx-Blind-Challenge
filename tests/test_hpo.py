"""Unit tests for HPO orchestrator module."""

from pathlib import Path

import pytest

from admet.model.chemprop.hpo import ChempropHPO, _flatten_dict
from admet.model.chemprop.hpo_config import (
    ASHAConfig,
    HPOConfig,
    ParameterSpace,
    ResourceConfig,
    SearchSpaceConfig,
)


class TestFlattenDict:
    """Tests for _flatten_dict helper function."""

    def test_empty_dict(self) -> None:
        """Test flattening empty dictionary."""
        result = _flatten_dict({})
        assert result == {}

    def test_flat_dict(self) -> None:
        """Test already flat dictionary."""
        d = {"a": 1, "b": "hello", "c": 3.14}
        result = _flatten_dict(d)
        assert result == {"a": 1, "b": "hello", "c": 3.14}

    def test_nested_dict(self) -> None:
        """Test flattening nested dictionary."""
        d = {"a": {"b": {"c": 1}}}
        result = _flatten_dict(d)
        assert result == {"a.b.c": 1}

    def test_mixed_dict(self) -> None:
        """Test flattening mixed nested/flat dictionary."""
        d = {
            "top": "value",
            "nested": {
                "key1": 1,
                "key2": 2,
            },
        }
        result = _flatten_dict(d)
        assert result == {
            "top": "value",
            "nested.key1": 1,
            "nested.key2": 2,
        }

    def test_custom_separator(self) -> None:
        """Test custom separator."""
        d = {"a": {"b": 1}}
        result = _flatten_dict(d, sep="_")
        assert result == {"a_b": 1}

    def test_non_primitive_converted_to_string(self) -> None:
        """Test that non-primitive values are converted to string."""
        d = {"list": [1, 2, 3]}
        result = _flatten_dict(d)
        assert result == {"list": "[1, 2, 3]"}


@pytest.fixture
def test_hpo_config() -> HPOConfig:
    """Create a test HPO configuration."""
    return HPOConfig(
        experiment_name="test_hpo",
        data_path="train.csv",
        val_data_path="validation.csv",
        smiles_column="smiles",
        target_columns=["target1", "target2"],
        output_dir="/tmp/hpo_test",
        seed=42,
        search_space=SearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
            dropout=ParameterSpace(type="uniform", low=0.0, high=0.4),
        ),
        asha=ASHAConfig(
            metric="val_mae",
            mode="min",
            max_t=50,
            grace_period=5,
        ),
        resources=ResourceConfig(
            num_samples=10,
            cpus_per_trial=1,
            gpus_per_trial=0,
        ),
    )


class TestChempropHPO:
    """Tests for ChempropHPO class."""

    def test_init(self, test_hpo_config) -> None:
        """Test HPO orchestrator initialization."""
        hpo = ChempropHPO(test_hpo_config)
        assert hpo.config == test_hpo_config
        assert hpo.results is None
        assert hpo._mlflow_run_id is None

    def test_build_search_space(self, test_hpo_config) -> None:
        """Test search space building."""
        hpo = ChempropHPO(test_hpo_config)
        space = hpo._build_search_space()

        # Should have tunable parameters
        assert "learning_rate" in space
        assert "dropout" in space

        # Should have fixed parameters (paths are converted to absolute)
        assert space["data_path"].endswith("train.csv")
        assert space["val_data_path"].endswith("validation.csv")
        assert space["smiles_column"] == "smiles"
        assert space["target_columns"] == ["target1", "target2"]
        assert space["max_epochs"] == 50
        assert space["metric"] == "val_mae"
        assert space["seed"] == 42

    def test_build_scheduler(self, test_hpo_config) -> None:
        """Test ASHA scheduler building."""
        hpo = ChempropHPO(test_hpo_config)
        scheduler = hpo._build_scheduler()

        assert scheduler is not None
        # The scheduler was built with the config values; just verify it's the right type
        # Ray's ASHAScheduler (AsyncHyperBandScheduler) internal attrs vary by version,
        # so we only check that the scheduler is instantiated correctly.
        from ray.tune.schedulers import ASHAScheduler

        assert isinstance(scheduler, ASHAScheduler)

    def test_setup_mlflow(self, test_hpo_config, mocker) -> None:
        """Test MLflow setup."""
        mock_mlflow = mocker.patch("admet.model.chemprop.hpo.mlflow")
        mock_run = mocker.MagicMock()
        mock_run.info.run_id = "test_run_id"
        # Setup context manager properly for 'with mlflow.start_run() as run:'
        mock_mlflow.start_run.return_value.__enter__ = mocker.MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = mocker.MagicMock(return_value=False)

        hpo = ChempropHPO(test_hpo_config)
        hpo._setup_mlflow()

        mock_mlflow.set_experiment.assert_called_once_with("test_hpo")
        mock_mlflow.start_run.assert_called_once()
        assert hpo._mlflow_run_id == "test_run_id"

    def test_get_top_k_configs_no_results(self, test_hpo_config) -> None:
        """Test get_top_k_configs with no results."""
        hpo = ChempropHPO(test_hpo_config)
        top_k = hpo.get_top_k_configs()
        assert top_k == []


class TestChempropHPOIntegration:
    """Integration tests for ChempropHPO (mocked)."""

    def test_run_creates_tuner(self, mocker) -> None:
        """Test that run() creates and runs a Ray Tune tuner."""
        # Setup mocks
        mock_mlflow = mocker.patch("admet.model.chemprop.hpo.mlflow")
        mock_tuner_class = mocker.patch("admet.model.chemprop.hpo.tune.Tuner")
        # Patch ray module where it's imported in the function body
        mocker.patch("ray.is_initialized", return_value=True)
        mocker.patch("ray.init")

        mock_run = mocker.MagicMock()
        mock_run.info.run_id = "test_run"
        mock_mlflow.start_run.return_value = mock_run

        mock_results = mocker.MagicMock()
        mock_results.get_best_result.return_value = None
        mock_results.get_dataframe.return_value = mocker.MagicMock()
        mock_results.get_dataframe.return_value.to_csv = mocker.MagicMock()
        mock_results.get_dataframe.return_value.columns = []
        mock_results.get_dataframe.return_value.sort_values.return_value = mock_results.get_dataframe.return_value
        mock_results.get_dataframe.return_value.head.return_value.iterrows.return_value = []

        mock_tuner = mocker.MagicMock()
        mock_tuner.fit.return_value = mock_results
        mock_tuner_class.return_value = mock_tuner

        config = HPOConfig(
            experiment_name="test",
            data_path="train.csv",
            target_columns=["target"],
            output_dir="/tmp/test_hpo",
        )
        hpo = ChempropHPO(config)

        mocker.patch("builtins.open", mocker.MagicMock())
        mocker.patch.object(Path, "mkdir")
        results = hpo.run()

        mock_tuner_class.assert_called_once()
        mock_tuner.fit.assert_called_once()
        assert results == mock_results
