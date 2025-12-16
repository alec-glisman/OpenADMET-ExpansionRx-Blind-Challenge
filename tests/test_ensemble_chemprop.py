"""
Tests for ChempropEnsemble configuration propagation and discovery helpers.
"""

import pytest

from admet.model.chemprop.config import CurriculumConfig, EnsembleConfig
from admet.model.chemprop.ensemble import ChempropEnsemble, SplitFoldInfo


@pytest.mark.no_mlflow_runs
def test_create_single_model_config_passes_curriculum(tmp_path):
    # Create a fake data structure that the ensemble expects
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    # Simple minimal EnsembleConfig with curriculum enabled
    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    # Create a curriculum setting
    ens_cfg.curriculum = CurriculumConfig(
        enabled=True,
        quality_col="Quality",
        qualities=["h", "m"],
        patience=2,
        seed=42,
    )

    # Disable MLflow in unit tests to avoid starting real runs
    ens_cfg.mlflow.tracking = False
    ensemble = ChempropEnsemble(ens_cfg)

    # Create dummy SplitFoldInfo
    info = SplitFoldInfo(
        split_idx=0,
        fold_idx=0,
        data_dir=data_dir,
        train_file=data_dir / "train.csv",
        validation_file=data_dir / "validation.csv",
    )

    model_cfg = ensemble._create_single_model_config(info)

    assert model_cfg.curriculum.enabled is True
    assert model_cfg.curriculum.quality_col == "Quality"
    assert model_cfg.curriculum.qualities == ["h", "m"]
    assert model_cfg.curriculum.patience == 2
    assert model_cfg.curriculum.seed == 42


def test_discover_splits_folds_sorted(tmp_path):
    # Create a fake data structure with unsorted split/fold directory names
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    splits = ["split_10", "split_2", "split_1"]
    folds = ["fold_2", "fold_10", "fold_1"]

    for s in splits:
        split_dir = data_dir / s
        split_dir.mkdir()
        for f in folds:
            fold_dir = split_dir / f
            fold_dir.mkdir(parents=True)
            # Create expected files
            (fold_dir / "train.csv").write_text("a,b,c\n1,2,3\n")
            (fold_dir / "validation.csv").write_text("a,b,c\n4,5,6\n")

    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    # Disable MLflow for tests
    ens_cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(ens_cfg)
    infos = ensemble.discover_splits_folds()

    # Expected order is numeric: split 1, then 2, then 10; folds 1,2,10
    expected = []
    for split_idx in [1, 2, 10]:
        for fold_idx in [1, 2, 10]:
            expected.append((split_idx, fold_idx))

    discovered = [(info.split_idx, info.fold_idx) for info in infos]

    assert discovered == expected


def test_discover_splits_folds_sorted_with_filters(tmp_path):
    # Create a fake data structure with unsorted split/fold directories
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    splits = ["split_10", "split_2", "split_1"]
    folds = ["fold_2", "fold_10", "fold_1"]

    for s in splits:
        split_dir = data_dir / s
        split_dir.mkdir()
        for f in folds:
            fold_dir = split_dir / f
            fold_dir.mkdir(parents=True)
            (fold_dir / "train.csv").write_text("a,b,c\n1,2,3\n")
            (fold_dir / "validation.csv").write_text("a,b,c\n4,5,6\n")

    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    # Provide filters unordered - discovery should sort them numerically
    ens_cfg.data.splits = [2, 1]
    ens_cfg.data.folds = [10, 2]
    ens_cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(ens_cfg)
    infos = ensemble.discover_splits_folds()

    # Expected: splits 1 then 2 (numeric), folds 2 then 10 (numeric), filtered
    expected = [(1, 2), (1, 10), (2, 2), (2, 10)]
    discovered = [(info.split_idx, info.fold_idx) for info in infos]

    assert discovered == expected


def test_create_single_model_config_includes_inter_task_affinity(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    ens_cfg = EnsembleConfig()
    ens_cfg.data.data_dir = str(data_dir)
    ens_cfg.data.target_cols = ["LogD"]
    ens_cfg.inter_task_affinity.enabled = True
    ens_cfg.inter_task_affinity.log_to_mlflow = True
    ens_cfg.inter_task_affinity.compute_every_n_steps = 2
    # Skip MLflow initialization for unit test
    ens_cfg.mlflow.tracking = False

    ensemble = ChempropEnsemble(ens_cfg)

    info = SplitFoldInfo(
        split_idx=0,
        fold_idx=0,
        data_dir=data_dir,
        train_file=data_dir / "train.csv",
        validation_file=data_dir / "validation.csv",
    )

    cfg = ensemble._create_single_model_config(info)
    assert cfg.inter_task_affinity.enabled is True
    assert cfg.inter_task_affinity.log_to_mlflow is True
    assert cfg.inter_task_affinity.compute_every_n_steps == 2


# Tests for plot metrics logging functionality


def test_sanitize_metric_label():
    """Test label sanitization utility function."""
    from admet.model.chemprop.ensemble import _sanitize_metric_label

    assert _sanitize_metric_label("LogD") == "logd"
    assert _sanitize_metric_label("Log KSOL") == "log_ksol"
    assert _sanitize_metric_label("Spearman $\\rho$") == "spearman_rho"
    assert _sanitize_metric_label("Kendall $\\tau$") == "kendall_tau"
    assert _sanitize_metric_label("$R^2$") == "r2"
    assert _sanitize_metric_label("Clearance > 5") == "clearance_gt_5"
    assert _sanitize_metric_label("Dose-Response") == "dose_response"


@pytest.mark.no_mlflow_runs
def test_log_plot_metrics_batch_logging(tmp_path):
    """Test that _log_plot_metrics uses batch logging correctly."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    # Create minimal config
    config = EnsembleConfig()
    config.data.data_dir = str(tmp_path)
    config.data.target_cols = ["LogD", "KSOL"]
    config.mlflow.tracking = False

    ensemble = ChempropEnsemble(config)

    # Mock MLflow client and parent run
    ensemble._mlflow_client = MagicMock()
    ensemble.parent_run_id = "test_run_123"

    # Mock mlflow.log_metrics
    with patch("admet.model.chemprop.ensemble.mlflow.log_metrics") as mock_log_metrics:
        # Call the method
        labels = ["logd", "ksol", "mean"]
        means = [0.45, 0.52, 0.485]
        errors = [0.03, 0.04, 0.05]

        ensemble._log_plot_metrics(
            split_name="test",
            metric_type="MAE",
            safe_metric="MAE",
            labels=labels,
            means=means,
            errors=errors,
            n_models=12,
        )

        # Verify batch logging was called once
        assert mock_log_metrics.call_count == 1

        # Verify the metrics dictionary
        call_args = mock_log_metrics.call_args[0][0]
        assert isinstance(call_args, dict)

        # Check expected keys
        assert "plots/test/MAE/logd" in call_args
        assert "plots/test/MAE/logd_stderr" in call_args
        assert "plots/test/MAE/ksol" in call_args
        assert "plots/test/MAE/ksol_stderr" in call_args
        assert "plots/test/MAE/mean" in call_args
        assert "plots/test/MAE/mean_stderr" in call_args
        assert "plots/test/MAE/n_models" in call_args

        # Check values
        assert call_args["plots/test/MAE/logd"] == 0.45
        assert call_args["plots/test/MAE/logd_stderr"] == 0.03
        assert call_args["plots/test/MAE/n_models"] == 12.0


@pytest.mark.no_mlflow_runs
def test_log_plot_metrics_handles_nan(tmp_path):
    """Test that NaN values are skipped with warnings."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    config = EnsembleConfig()
    config.data.data_dir = str(tmp_path)
    config.data.target_cols = ["LogD"]
    config.mlflow.tracking = False

    ensemble = ChempropEnsemble(config)
    ensemble._mlflow_client = MagicMock()
    ensemble.parent_run_id = "test_run_123"

    with patch("admet.model.chemprop.ensemble.mlflow.log_metrics") as mock_log_metrics:
        # Include a NaN value
        labels = ["logd", "ksol", "mean"]
        means = [0.45, np.nan, 0.485]
        errors = [0.03, 0.04, 0.05]

        ensemble._log_plot_metrics(
            split_name="test",
            metric_type="MAE",
            safe_metric="MAE",
            labels=labels,
            means=means,
            errors=errors,
            n_models=3,
        )

        # Verify batch logging was still called
        assert mock_log_metrics.call_count == 1

        # Verify NaN value was skipped
        call_args = mock_log_metrics.call_args[0][0]
        assert "plots/test/MAE/logd" in call_args
        assert "plots/test/MAE/ksol" not in call_args  # Skipped due to NaN
        assert "plots/test/MAE/mean" in call_args
