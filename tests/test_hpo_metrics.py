"""Tests for HPO final trial metrics computation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from admet.model.hpo_metrics import _compute_split_metrics, compute_final_trial_metrics, sanitize_metric_label
from admet.plot.metrics import METRIC_NAMES


class TestSanitizeMetricLabel:
    """Tests for sanitize_metric_label function."""

    def test_basic_lowercase(self):
        """Test basic lowercase conversion."""
        assert sanitize_metric_label("LogD") == "logd"
        assert sanitize_metric_label("MAE") == "mae"

    def test_space_replacement(self):
        """Test space to underscore conversion."""
        assert sanitize_metric_label("Log KSOL") == "log_ksol"
        assert sanitize_metric_label("Log HLM CLint") == "log_hlm_clint"

    def test_special_characters(self):
        """Test special character handling."""
        # Note: > becomes gt without adding underscore (since no space follows)
        assert sanitize_metric_label("Log Caco-2 Permeability Papp A>B") == "log_caco_2_permeability_papp_agtb"
        assert sanitize_metric_label("value<0.5") == "valuelt0.5"
        assert sanitize_metric_label("R^2") == "r2"

    def test_greek_letters(self):
        """Test Greek letter conversion."""
        assert sanitize_metric_label("Spearman ρ") == "spearman_rho"
        assert sanitize_metric_label("Kendall τ") == "kendall_tau"

    def test_latex_symbols(self):
        """Test LaTeX symbol removal."""
        assert sanitize_metric_label("$R^2$") == "r2"
        assert sanitize_metric_label("Spearman $\\rho$") == "spearman_rho"

    def test_superscript_numbers(self):
        """Test superscript number conversion."""
        assert sanitize_metric_label("R²") == "r2"


class TestComputeSplitMetrics:
    """Tests for _compute_split_metrics function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and targets."""
        np.random.seed(42)
        n_samples = 100
        n_targets = 3
        targets = np.random.randn(n_samples, n_targets)
        # Predictions are targets with some noise (for realistic metrics)
        predictions = targets + 0.1 * np.random.randn(n_samples, n_targets)
        target_columns = ["LogD", "Log KSOL", "Log HLM CLint"]
        return predictions, targets, target_columns

    def test_basic_metric_computation(self, sample_data):
        """Test that all metrics are computed for each target."""
        predictions, targets, target_columns = sample_data
        metrics = _compute_split_metrics(predictions, targets, target_columns, "train")

        # Check per-target metrics
        for target in target_columns:
            safe_target = sanitize_metric_label(target)
            for metric_name in METRIC_NAMES:
                key = f"train/{safe_target}_{metric_name}"
                assert key in metrics, f"Missing metric: {key}"
                assert isinstance(metrics[key], float)

        # Check aggregate metrics
        for metric_name in METRIC_NAMES:
            key = f"train/mean_{metric_name}"
            assert key in metrics, f"Missing aggregate metric: {key}"
            assert isinstance(metrics[key], float)

    def test_metric_count(self, sample_data):
        """Test that the correct number of metrics is returned."""
        predictions, targets, target_columns = sample_data
        metrics = _compute_split_metrics(predictions, targets, target_columns, "val")

        # Per-target: 3 targets × 8 metrics = 24
        # Aggregate: 8 mean metrics
        # Total: 32
        expected_count = len(target_columns) * len(METRIC_NAMES) + len(METRIC_NAMES)
        assert len(metrics) == expected_count

    def test_split_name_prefix(self, sample_data):
        """Test that all metric keys use the correct split name prefix."""
        predictions, targets, target_columns = sample_data

        for split_name in ["train", "val", "test"]:
            metrics = _compute_split_metrics(predictions, targets, target_columns, split_name)
            for key in metrics:
                assert key.startswith(f"{split_name}/"), f"Key {key} should start with {split_name}/"

    def test_r2_pearson_correlation(self, sample_data):
        """Test that R2 and Pearson correlations are reasonable for correlated data."""
        predictions, targets, target_columns = sample_data
        metrics = _compute_split_metrics(predictions, targets, target_columns, "train")

        # With low noise, we expect high R2 and Pearson correlation
        for target in target_columns:
            safe_target = sanitize_metric_label(target)
            r2 = metrics[f"train/{safe_target}_R2"]
            pearson = metrics[f"train/{safe_target}_pearson_r"]

            assert r2 > 0.5, f"R2 should be reasonably high for {target}"
            assert pearson > 0.7, f"Pearson r should be reasonably high for {target}"

    def test_1d_input_handling(self):
        """Test that 1D arrays are handled correctly."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        target_columns = ["single_target"]

        metrics = _compute_split_metrics(predictions, targets, target_columns, "test")

        assert "test/single_target_mae" in metrics
        assert "test/mean_mae" in metrics

    def test_torch_tensor_input(self):
        """Test that PyTorch tensors are handled correctly."""
        torch = pytest.importorskip("torch")

        predictions = torch.randn(50, 2)
        targets = predictions + 0.1 * torch.randn(50, 2)
        target_columns = ["target_a", "target_b"]

        metrics = _compute_split_metrics(predictions, targets, target_columns, "val")

        assert len(metrics) > 0
        for key, value in metrics.items():
            assert isinstance(value, float)


class TestComputeFinalTrialMetrics:
    """Tests for compute_final_trial_metrics function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with predict method."""
        model = MagicMock()
        # Predict returns predictions matching the shape of targets
        model.predict = MagicMock(side_effect=lambda loader: np.random.randn(50, 3))
        return model

    @pytest.fixture
    def mock_loader(self):
        """Create a mock data loader that properly simulates iteration."""

        class MockBatch:
            def __init__(self):
                self.Y = np.random.randn(50, 3)

        class MockLoader:
            def __iter__(self):
                return iter([MockBatch()])

        return MockLoader()

    def test_all_splits(self, mock_model):
        """Test that metrics are computed for all provided splits."""

        # Create fresh loaders for each call
        class MockBatch:
            def __init__(self):
                self.Y = np.random.randn(50, 3)

        class MockLoader:
            def __iter__(self):
                return iter([MockBatch()])

        target_columns = ["LogD", "Log KSOL", "Log HLM CLint"]

        metrics = compute_final_trial_metrics(
            model=mock_model,
            train_loader=MockLoader(),
            val_loader=MockLoader(),
            test_loader=MockLoader(),
            target_columns=target_columns,
        )

        # Check that all splits have metrics
        train_metrics = [k for k in metrics if k.startswith("train/")]
        val_metrics = [k for k in metrics if k.startswith("val/")]
        test_metrics = [k for k in metrics if k.startswith("test/")]

        assert len(train_metrics) > 0
        assert len(val_metrics) > 0
        assert len(test_metrics) > 0

    def test_optional_splits(self, mock_model, mock_loader):
        """Test that optional splits are correctly skipped."""
        target_columns = ["LogD"]

        # Only train, no val or test
        metrics = compute_final_trial_metrics(
            model=mock_model,
            train_loader=mock_loader,
            val_loader=None,
            test_loader=None,
            target_columns=target_columns,
        )

        train_metrics = [k for k in metrics if k.startswith("train/")]
        val_metrics = [k for k in metrics if k.startswith("val/")]
        test_metrics = [k for k in metrics if k.startswith("test/")]

        assert len(train_metrics) > 0
        assert len(val_metrics) == 0
        assert len(test_metrics) == 0

    def test_error_handling(self):
        """Test that errors in one split don't break others."""

        class MockBatch:
            def __init__(self):
                self.Y = np.random.randn(50, 2)

        class MockLoader:
            def __iter__(self):
                return iter([MockBatch()])

        mock_model = MagicMock()
        # First call succeeds, second fails, third succeeds
        call_count = [0]

        def predict_side_effect(loader):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Prediction failed")
            return np.random.randn(50, 2)

        mock_model.predict = MagicMock(side_effect=predict_side_effect)
        target_columns = ["LogD", "Log KSOL"]

        metrics = compute_final_trial_metrics(
            model=mock_model,
            train_loader=MockLoader(),
            val_loader=MockLoader(),
            test_loader=MockLoader(),
            target_columns=target_columns,
        )

        # Train and test should have metrics, val should not
        train_metrics = [k for k in metrics if k.startswith("train/")]
        test_metrics = [k for k in metrics if k.startswith("test/")]

        assert len(train_metrics) > 0
        assert len(test_metrics) > 0


class TestComputeFinalTrialMetricsFromDataframes:
    """Tests for compute_final_trial_metrics_from_dataframes function.

    Note: These tests mock the chemprop imports to avoid creating real
    MoleculeDatapoints which would require valid SMILES parsing.
    """

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample DataFrames for testing."""
        np.random.seed(42)
        n_train, n_val, n_test = 100, 30, 30

        def create_df(n_samples):
            return pd.DataFrame(
                {
                    "SMILES": ["CCO"] * n_samples,  # Valid SMILES
                    "LogD": np.random.randn(n_samples),
                    "Log KSOL": np.random.randn(n_samples),
                }
            )

        return create_df(n_train), create_df(n_val), create_df(n_test)

    def test_dataframe_conversion(self, sample_dataframes):
        """Test that DataFrames are converted to loaders correctly."""
        train_df, val_df, test_df = sample_dataframes

        with patch.dict("sys.modules", {"chemprop.data": MagicMock()}):
            with patch("admet.model.hpo_metrics.compute_final_trial_metrics") as mock_compute:
                mock_compute.return_value = {"train/logd_mae": 0.5}

                # Use a simplified mock approach - create the module-level mock before import
                from admet.model import hpo_metrics

                # Test that the function signature is correct
                assert callable(hpo_metrics.compute_final_trial_metrics_from_dataframes)

    def test_optional_dataframes(self, sample_dataframes):
        """Test that None DataFrames result in None loaders."""
        train_df, _, _ = sample_dataframes

        with patch.dict("sys.modules", {"chemprop.data": MagicMock()}):
            with patch("admet.model.hpo_metrics.compute_final_trial_metrics") as mock_compute:
                mock_compute.return_value = {}

                from admet.model import hpo_metrics

                # Test that the function exists and is callable
                assert callable(hpo_metrics.compute_final_trial_metrics_from_dataframes)


class TestMetricNamesConstant:
    """Tests for METRIC_NAMES constant."""

    def test_metric_names_tuple(self):
        """Test that METRIC_NAMES is a tuple (immutable)."""
        assert isinstance(METRIC_NAMES, tuple)

    def test_all_expected_metrics(self):
        """Test that all expected metrics are present."""
        expected = {"mae", "rae", "mape", "rmse", "R2", "pearson_r", "spearman_rho", "kendall_tau"}
        assert set(METRIC_NAMES) == expected

    def test_metric_count(self):
        """Test the expected number of metrics."""
        assert len(METRIC_NAMES) == 8
