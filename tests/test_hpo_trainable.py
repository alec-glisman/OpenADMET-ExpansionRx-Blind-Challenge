"""Unit tests for HPO trainable module."""

import numpy as np
import torch

from admet.model.chemprop.hpo_trainable import (
    RayTuneReportCallback,
    _build_hyperparams,
    _extract_target_weights,
)
from admet.model.chemprop.model import ChempropHyperparams


class TestRayTuneReportCallback:
    """Tests for RayTuneReportCallback class."""

    def test_init_default_metric(self) -> None:
        """Test default metric is val_mae."""
        callback = RayTuneReportCallback()
        assert callback.metric == "val_mae"

    def test_init_custom_metric(self) -> None:
        """Test custom metric initialization."""
        callback = RayTuneReportCallback(metric="val_loss")
        assert callback.metric == "val_loss"

    def test_on_validation_end_reports_metrics(self, mocker) -> None:
        """Test that validation end reports metrics to Ray Tune."""
        mock_report = mocker.patch("admet.model.chemprop.hpo_trainable.train.report")

        callback = RayTuneReportCallback(metric="val_mae")

        # Create mock trainer
        mock_trainer = mocker.MagicMock()
        mock_trainer.callback_metrics = {
            "val_mae": torch.tensor(0.5),
            "train_loss": torch.tensor(0.3),
        }
        mock_trainer.current_epoch = 10

        # Create mock pl_module
        mock_pl_module = mocker.MagicMock()

        # Call the callback
        callback.on_validation_end(mock_trainer, mock_pl_module)

        # Verify report was called
        mock_report.assert_called_once()
        call_args = mock_report.call_args[0][0]
        assert "val_mae" in call_args
        assert np.isclose(call_args["val_mae"], 0.5, atol=1e-6)
        assert call_args["epoch"] == 10
        assert np.isclose(call_args["train_loss"], 0.3, atol=1e-6)

    def test_on_validation_end_skips_empty_metrics(self, mocker) -> None:
        """Test that empty metrics are skipped."""
        mock_report = mocker.patch("admet.model.chemprop.hpo_trainable.train.report")

        callback = RayTuneReportCallback()

        mock_trainer = mocker.MagicMock()
        mock_trainer.callback_metrics = {}

        callback.on_validation_end(mock_trainer, mocker.MagicMock())
        mock_report.assert_not_called()


class TestBuildHyperparams:
    """Tests for _build_hyperparams function."""

    def test_minimal_config(self) -> None:
        """Test building hyperparams with minimal config."""
        config: dict[str, object] = {}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert isinstance(params, ChempropHyperparams)
        assert params.max_epochs == 100
        assert params.seed == 42

    def test_learning_rate_mapping(self) -> None:
        """Test that learning_rate maps to lr parameters."""
        config = {"learning_rate": 0.001}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.max_lr == 0.001
        assert params.init_lr == 0.0001  # 1/10 of max_lr
        assert params.final_lr == 0.0001

    def test_dropout_mapping(self) -> None:
        """Test dropout parameter mapping."""
        config = {"dropout": 0.3}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.dropout == 0.3

    def test_depth_mapping(self) -> None:
        """Test depth parameter mapping."""
        config = {"depth": 5}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.depth == 5

    def test_hidden_dim_mapping(self) -> None:
        """Test hidden_dim maps to both message_hidden_dim and hidden_dim."""
        config = {"hidden_dim": 512}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.message_hidden_dim == 512
        assert params.hidden_dim == 512

    def test_ffn_parameters_mapping(self) -> None:
        """Test FFN parameter mapping."""
        config = {
            "ffn_num_layers": 3,
            "ffn_hidden_dim": 256,
        }
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.num_layers == 3
        assert params.hidden_dim == 256

    def test_batch_size_mapping(self) -> None:
        """Test batch_size parameter mapping."""
        config = {"batch_size": 64}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.batch_size == 64

    def test_ffn_type_mapping_mlp(self) -> None:
        """Test FFN type mapping for MLP."""
        config = {"ffn_type": "mlp"}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.ffn_type == "regression"

    def test_ffn_type_mapping_moe(self) -> None:
        """Test FFN type mapping for MoE."""
        config = {"ffn_type": "moe"}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.ffn_type == "mixture_of_experts"

    def test_ffn_type_mapping_branched(self) -> None:
        """Test FFN type mapping for branched."""
        config = {"ffn_type": "branched"}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.ffn_type == "branched"

    def test_n_experts_mapping(self) -> None:
        """Test n_experts parameter mapping."""
        config = {"n_experts": 8}
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.n_experts == 8

    def test_trunk_parameters_mapping(self) -> None:
        """Test trunk parameter mapping for branched FFN."""
        config = {
            "trunk_depth": 2,
            "trunk_hidden_dim": 300,
        }
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        assert params.trunk_n_layers == 2
        assert params.trunk_hidden_dim == 300

    def test_none_values_ignored(self) -> None:
        """Test that None values are ignored."""
        config = {
            "learning_rate": None,
            "dropout": 0.2,
        }
        params = _build_hyperparams(config, max_epochs=100, seed=42)
        # Learning rate should be default
        assert params.dropout == 0.2


class TestExtractTargetWeights:
    """Tests for _extract_target_weights function."""

    def test_extracts_weights_for_all_targets(self) -> None:
        """Test that weights are extracted for all targets."""
        config = {
            "target_weight_LogD": 1.5,
            "target_weight_Log_KSOL": 2.0,
            "target_weight_Log_HLM_CLint": 0.5,
        }
        target_columns = ["LogD", "Log KSOL", "Log HLM CLint"]
        weights = _extract_target_weights(config, target_columns)

        assert len(weights) == 3
        assert weights[0] == 1.5  # LogD
        assert weights[1] == 2.0  # Log KSOL
        assert weights[2] == 0.5  # Log HLM CLint

    def test_defaults_to_one_for_missing_weights(self) -> None:
        """Test that missing weights default to 1.0."""
        config = {
            "target_weight_LogD": 2.0,
            # Log KSOL missing
        }
        target_columns = ["LogD", "Log KSOL"]
        weights = _extract_target_weights(config, target_columns)

        assert len(weights) == 2
        assert weights[0] == 2.0  # LogD - from config
        assert weights[1] == 1.0  # Log KSOL - default

    def test_handles_special_characters_in_names(self) -> None:
        """Test that special characters are handled correctly."""
        config = {
            "target_weight_Log_Caco-2_Permeability_Papp_AgtB": 3.0,
        }
        target_columns = ["Log Caco-2 Permeability Papp A>B"]
        weights = _extract_target_weights(config, target_columns)

        assert len(weights) == 1
        assert weights[0] == 3.0

    def test_handles_none_values(self) -> None:
        """Test that None values are converted to 1.0."""
        config = {
            "target_weight_LogD": None,
        }
        target_columns = ["LogD"]
        weights = _extract_target_weights(config, target_columns)

        assert weights[0] == 1.0

    def test_empty_target_columns(self) -> None:
        """Test with empty target columns."""
        config = {"target_weight_LogD": 2.0}
        target_columns: list[str] = []
        weights = _extract_target_weights(config, target_columns)

        assert len(weights) == 0
