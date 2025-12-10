"""
Tests for task affinity grouping module.

This module tests the Task Affinity Grouping (TAG) algorithm implementation
for multi-task learning with Chemprop models.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from chemprop import models, nn as chemprop_nn

from admet.model.chemprop.task_affinity import (
    TaskAffinityComputer,
    TaskAffinityConfig,
    TaskGrouper,
    _flatten_gradients,
    _get_device,
    _is_encoder_param,
    _masked_mse_loss,
    affinity_matrix_to_dataframe,
    compute_task_affinity,
    get_task_group_indices,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "C1=CC=C(C=C1)CC(C(=O)O)N",  # Phenylalanine
        "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
        "CCO",  # Ethanol
        "CCCC",  # Butane
        "C1=CC=CC=C1",  # Benzene
    ]


@pytest.fixture
def sample_targets():
    """Sample target values for testing (with some NaN values)."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [1.5, np.nan, 3.5],
            [2.0, 2.5, np.nan],
            [2.5, 3.0, 4.5],
            [np.nan, 3.5, 5.0],
            [1.2, 2.2, 3.2],
            [1.8, 2.8, 3.8],
            [2.2, 3.2, 4.2],
        ]
    )


@pytest.fixture
def sample_dataframe(sample_smiles, sample_targets):
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "SMILES": sample_smiles,
            "LogD": sample_targets[:, 0],
            "KSOL": sample_targets[:, 1],
            "PAMPA": sample_targets[:, 2],
        }
    )


@pytest.fixture
def task_affinity_config():
    """Default task affinity configuration."""
    return TaskAffinityConfig(
        enabled=True,
        affinity_epochs=1,
        affinity_batch_size=4,
        affinity_lr=1e-3,
        n_groups=2,
        clustering_method="agglomerative",
        affinity_type="cosine",
        seed=42,
    )


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================


class TestGetDevice:
    """Tests for _get_device function."""

    def test_auto_device(self):
        """Test auto device selection."""
        device = _get_device("auto")
        assert device.type in ["cpu", "cuda"]

    def test_cpu_device(self):
        """Test explicit CPU device."""
        device = _get_device("cpu")
        assert device.type == "cpu"

    def test_cuda_device(self):
        """Test explicit CUDA device (may fallback to CPU if not available)."""
        device = _get_device("cuda")
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            # torch.device("cuda") is valid even without GPU
            assert device.type == "cuda"


class TestIsEncoderParam:
    """Tests for _is_encoder_param function."""

    def test_encoder_param_default(self):
        """Test default encoder parameter detection."""
        # These should be encoder params
        assert _is_encoder_param("message_passing.weight", []) is True
        assert _is_encoder_param("encoder.layer1.weight", []) is True
        assert _is_encoder_param("mp.weight", []) is True

    def test_predictor_param_default(self):
        """Test predictor parameter exclusion."""
        # These should NOT be encoder params
        assert _is_encoder_param("predictor.weight", []) is False
        assert _is_encoder_param("ffn.layer1.weight", []) is False
        assert _is_encoder_param("output.bias", []) is False
        assert _is_encoder_param("readout.weight", []) is False

    def test_custom_patterns(self):
        """Test custom pattern matching."""
        patterns = ["encoder", "mp"]
        assert _is_encoder_param("encoder.weight", patterns) is True
        assert _is_encoder_param("mp.bias", patterns) is True
        assert _is_encoder_param("ffn.weight", patterns) is False


class TestMaskedMseLoss:
    """Tests for _masked_mse_loss function."""

    def test_no_nan_values(self):
        """Test MSE with no NaN values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, 2.1, 3.1])
        loss = _masked_mse_loss(pred, target)
        assert loss is not None
        assert loss.item() == pytest.approx(0.01, rel=1e-4)

    def test_with_nan_values(self):
        """Test MSE with NaN values in target."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.1, float("nan"), 3.1])
        loss = _masked_mse_loss(pred, target)
        assert loss is not None
        # Only 2 valid entries
        expected = ((1.0 - 1.1) ** 2 + (3.0 - 3.1) ** 2) / 2
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_all_nan_values(self):
        """Test MSE when all targets are NaN."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([float("nan"), float("nan"), float("nan")])
        loss = _masked_mse_loss(pred, target)
        assert loss is None


class TestFlattenGradients:
    """Tests for _flatten_gradients function."""

    def test_flatten_single_tensor(self):
        """Test flattening a single gradient tensor."""
        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _flatten_gradients((grad,), torch.device("cpu"))
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_flatten_multiple_tensors(self):
        """Test flattening multiple gradient tensors."""
        grad1 = torch.tensor([1.0, 2.0])
        grad2 = torch.tensor([3.0, 4.0])
        result = _flatten_gradients((grad1, grad2), torch.device("cpu"))
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_flatten_with_none(self):
        """Test flattening with None gradients."""
        grad1 = torch.tensor([1.0, 2.0])
        grad2 = None
        grad3 = torch.tensor([3.0])
        result = _flatten_gradients((grad1, grad2, grad3), torch.device("cpu"))
        # None is replaced with zeros(1)
        expected = np.array([1.0, 2.0, 0.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Unit Tests for TaskAffinityConfig
# =============================================================================


class TestTaskAffinityConfig:
    """Tests for TaskAffinityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TaskAffinityConfig()
        assert config.enabled is False
        assert config.affinity_epochs == 1
        assert config.affinity_batch_size == 64
        assert config.n_groups == 3
        assert config.clustering_method == "agglomerative"
        assert config.affinity_type == "cosine"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TaskAffinityConfig(
            enabled=True,
            affinity_epochs=5,
            n_groups=4,
            clustering_method="spectral",
        )
        assert config.enabled is True
        assert config.affinity_epochs == 5
        assert config.n_groups == 4
        assert config.clustering_method == "spectral"


# =============================================================================
# Unit Tests for TaskGrouper
# =============================================================================


class TestTaskGrouper:
    """Tests for TaskGrouper class."""

    def test_agglomerative_clustering(self):
        """Test agglomerative clustering."""
        # Create a simple affinity matrix where tasks 0,1 are similar
        # and tasks 2,3 are similar
        affinity = np.array(
            [
                [1.0, 0.9, 0.1, 0.1],
                [0.9, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.9],
                [0.1, 0.1, 0.9, 1.0],
            ]
        )
        task_names = ["Task0", "Task1", "Task2", "Task3"]

        grouper = TaskGrouper(n_groups=2, method="agglomerative")
        groups = grouper.cluster(affinity, task_names)

        assert len(groups) == 2
        assert grouper.labels is not None
        assert len(grouper.labels) == 4

        # Check that similar tasks are in the same group
        group_for_task0 = grouper.labels[0]
        group_for_task1 = grouper.labels[1]
        group_for_task2 = grouper.labels[2]
        group_for_task3 = grouper.labels[3]

        assert group_for_task0 == group_for_task1
        assert group_for_task2 == group_for_task3
        assert group_for_task0 != group_for_task2

    def test_spectral_clustering(self):
        """Test spectral clustering."""
        affinity = np.array(
            [
                [1.0, 0.9, 0.1, 0.1],
                [0.9, 1.0, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.9],
                [0.1, 0.1, 0.9, 1.0],
            ]
        )
        task_names = ["Task0", "Task1", "Task2", "Task3"]

        grouper = TaskGrouper(n_groups=2, method="spectral", seed=42)
        groups = grouper.cluster(affinity, task_names)

        assert len(groups) == 2
        assert grouper.labels is not None

    def test_more_groups_than_tasks(self):
        """Test when n_groups >= n_tasks."""
        affinity = np.array([[1.0, 0.5], [0.5, 1.0]])
        task_names = ["Task0", "Task1"]

        grouper = TaskGrouper(n_groups=3)
        groups = grouper.cluster(affinity, task_names)

        # Should put one task per group
        assert len(groups) == 2
        assert groups == [["Task0"], ["Task1"]]

    def test_invalid_method(self):
        """Test invalid clustering method."""
        grouper = TaskGrouper(n_groups=2, method="invalid")
        affinity = np.array([[1.0, 0.5], [0.5, 1.0]])

        with pytest.raises(ValueError, match="Unknown clustering method"):
            grouper.cluster(affinity, ["Task0", "Task1"])


# =============================================================================
# Unit Tests for Utility Functions
# =============================================================================


class TestGetTaskGroupIndices:
    """Tests for get_task_group_indices function."""

    def test_simple_case(self):
        """Test converting task groups to indices."""
        groups = [["LogD", "KSOL"], ["PAMPA", "hERG"]]
        target_cols = ["LogD", "KSOL", "PAMPA", "hERG"]
        indices = get_task_group_indices(groups, target_cols)
        assert indices == [[0, 1], [2, 3]]

    def test_reordered_tasks(self):
        """Test with reordered task names."""
        groups = [["PAMPA"], ["LogD", "hERG"], ["KSOL"]]
        target_cols = ["LogD", "KSOL", "PAMPA", "hERG"]
        indices = get_task_group_indices(groups, target_cols)
        assert indices == [[2], [0, 3], [1]]


class TestAffinityMatrixToDataframe:
    """Tests for affinity_matrix_to_dataframe function."""

    def test_conversion(self):
        """Test converting affinity matrix to DataFrame."""
        affinity = np.array([[1.0, 0.5], [0.5, 1.0]])
        task_names = ["Task0", "Task1"]

        df = affinity_matrix_to_dataframe(affinity, task_names)

        assert df.shape == (2, 2)
        assert list(df.index) == task_names
        assert list(df.columns) == task_names
        assert df.loc["Task0", "Task1"] == 0.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestTaskAffinityComputer:
    """Integration tests for TaskAffinityComputer."""

    def test_compute_from_dataframe(self, sample_dataframe, task_affinity_config):
        """Test computing affinity from DataFrame."""
        computer = TaskAffinityComputer(task_affinity_config)
        target_cols = ["LogD", "KSOL", "PAMPA"]

        affinity, task_names = computer.compute_from_dataframe(
            sample_dataframe,
            smiles_col="SMILES",
            target_cols=target_cols,
        )

        # Check output shape
        assert affinity.shape == (3, 3)
        assert task_names == target_cols

        # Affinity matrix should be symmetric
        np.testing.assert_array_almost_equal(affinity, affinity.T)

        # Diagonal should be close to 1 for cosine affinity
        for i in range(3):
            assert affinity[i, i] == pytest.approx(1.0, abs=0.2)

    def test_affinity_matrix_stored(self, sample_dataframe, task_affinity_config):
        """Test that affinity matrix is stored in computer."""
        computer = TaskAffinityComputer(task_affinity_config)
        target_cols = ["LogD", "KSOL", "PAMPA"]

        computer.compute_from_dataframe(
            sample_dataframe,
            smiles_col="SMILES",
            target_cols=target_cols,
        )

        assert computer.affinity_matrix is not None
        assert computer.task_names is not None


class TestComputeTaskAffinity:
    """Integration tests for compute_task_affinity function."""

    def test_full_pipeline(self, sample_dataframe):
        """Test the full task affinity computation pipeline."""
        config = TaskAffinityConfig(
            enabled=True,
            affinity_epochs=1,
            affinity_batch_size=4,
            n_groups=2,
            seed=42,
        )

        affinity, task_names, groups = compute_task_affinity(
            sample_dataframe,
            smiles_col="SMILES",
            target_cols=["LogD", "KSOL", "PAMPA"],
            config=config,
        )

        # Check affinity matrix
        assert affinity.shape == (3, 3)
        assert task_names == ["LogD", "KSOL", "PAMPA"]

        # Check groups
        assert len(groups) == 2
        # All tasks should be assigned to a group
        all_tasks = [t for g in groups for t in g]
        assert set(all_tasks) == {"LogD", "KSOL", "PAMPA"}

    def test_default_config(self, sample_dataframe):
        """Test with default configuration."""
        affinity, task_names, groups = compute_task_affinity(
            sample_dataframe,
            smiles_col="SMILES",
            target_cols=["LogD", "KSOL", "PAMPA"],
            config=None,  # Use defaults
        )

        assert affinity.shape == (3, 3)
        assert len(groups) == 3  # Default n_groups is 3


# =============================================================================
# Configuration Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Tests for TaskAffinityConfig integration with ChempropConfig."""

    def test_chemprop_config_has_task_affinity(self):
        """Test that ChempropConfig includes task_affinity field."""
        from admet.model.chemprop.config import ChempropConfig, TaskAffinityConfig

        config = ChempropConfig()
        assert hasattr(config, "task_affinity")
        assert isinstance(config.task_affinity, TaskAffinityConfig)
        assert config.task_affinity.enabled is False

    def test_chemprop_config_with_task_affinity_enabled(self):
        """Test ChempropConfig with task affinity enabled."""
        from admet.model.chemprop.config import ChempropConfig, TaskAffinityConfig

        config = ChempropConfig(
            task_affinity=TaskAffinityConfig(
                enabled=True,
                n_groups=3,
                affinity_epochs=2,
            )
        )

        assert config.task_affinity.enabled is True
        assert config.task_affinity.n_groups == 3
        assert config.task_affinity.affinity_epochs == 2

    def test_ensemble_config_has_task_affinity(self):
        """Test that EnsembleConfig includes task_affinity field."""
        from admet.model.chemprop.config import EnsembleConfig, TaskAffinityConfig

        config = EnsembleConfig()
        assert hasattr(config, "task_affinity")
        assert isinstance(config.task_affinity, TaskAffinityConfig)
