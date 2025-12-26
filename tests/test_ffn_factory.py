"""Tests for shared FFN factory."""

import pytest
import torch
from chemprop import nn

from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN
from admet.model.ffn_factory import create_ffn_predictor


class TestCreateFFNPredictor:
    """Tests for create_ffn_predictor factory function."""

    @pytest.mark.parametrize(
        "ffn_type,expected_class",
        [
            ("regression", nn.RegressionFFN),
            ("mixture_of_experts", MixtureOfExpertsRegressionFFN),
            ("branched", BranchedFFN),
        ],
    )
    def test_creates_correct_ffn_type(self, ffn_type: str, expected_class: type) -> None:
        """Test that factory creates the correct FFN class for each type."""
        ffn = create_ffn_predictor(
            ffn_type=ffn_type,
            input_dim=300,
            n_tasks=3,
        )
        assert isinstance(ffn, expected_class)

    def test_regression_ffn_parameters(self) -> None:
        """Test regression FFN is created with correct parameters."""
        ffn = create_ffn_predictor(
            ffn_type="regression",
            input_dim=256,
            n_tasks=5,
            hidden_dim=128,
            n_layers=3,
            dropout=0.1,
        )
        assert isinstance(ffn, nn.RegressionFFN)
        assert ffn.input_dim == 256
        assert ffn.n_tasks == 5

    def test_moe_ffn_parameters(self) -> None:
        """Test MoE FFN is created with correct parameters."""
        ffn = create_ffn_predictor(
            ffn_type="mixture_of_experts",
            input_dim=300,
            n_tasks=4,
            n_experts=6,
            hidden_dim=200,
        )
        assert isinstance(ffn, MixtureOfExpertsRegressionFFN)
        assert ffn.n_experts == 6
        assert ffn.n_tasks == 4

    def test_moe_default_n_experts(self) -> None:
        """Test MoE FFN uses default n_experts=4 when not specified."""
        ffn = create_ffn_predictor(
            ffn_type="mixture_of_experts",
            input_dim=300,
            n_tasks=2,
        )
        assert isinstance(ffn, MixtureOfExpertsRegressionFFN)
        assert ffn.n_experts == 4

    def test_branched_ffn_parameters(self) -> None:
        """Test branched FFN is created with correct parameters."""
        task_groups = [[0, 1], [2, 3]]
        ffn = create_ffn_predictor(
            ffn_type="branched",
            input_dim=300,
            n_tasks=4,
            task_groups=task_groups,
            trunk_n_layers=3,
            trunk_hidden_dim=400,
        )
        assert isinstance(ffn, BranchedFFN)
        assert ffn.n_tasks == 4
        assert ffn.task_groups == task_groups
        assert ffn.trunk_layers == 3

    def test_branched_default_task_groups(self) -> None:
        """Test branched FFN creates one task per group by default."""
        ffn = create_ffn_predictor(
            ffn_type="branched",
            input_dim=300,
            n_tasks=3,
        )
        assert isinstance(ffn, BranchedFFN)
        assert ffn.task_groups == [[0], [1], [2]]

    def test_invalid_ffn_type_raises_error(self) -> None:
        """Test that invalid ffn_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ffn_type"):
            create_ffn_predictor(
                ffn_type="invalid_type",
                input_dim=300,
                n_tasks=2,
            )

    def test_ffn_forward_pass(self) -> None:
        """Test that all FFN types can perform forward pass."""
        batch_size = 8
        input_dim = 300
        n_tasks = 3
        x = torch.randn(batch_size, input_dim)

        for ffn_type in ["regression", "mixture_of_experts", "branched"]:
            ffn = create_ffn_predictor(
                ffn_type=ffn_type,
                input_dim=input_dim,
                n_tasks=n_tasks,
            )
            output = ffn(x)
            assert output.shape == (batch_size, n_tasks)

    def test_with_task_weights(self) -> None:
        """Test FFN creation with task weights."""
        task_weights = torch.tensor([1.0, 2.0, 0.5])
        ffn = create_ffn_predictor(
            ffn_type="regression",
            input_dim=300,
            n_tasks=3,
            task_weights=task_weights,
        )
        assert isinstance(ffn, nn.RegressionFFN)

    def test_with_criterion(self) -> None:
        """Test FFN creation with custom criterion."""
        from chemprop.nn.metrics import MSE

        criterion = MSE()
        ffn = create_ffn_predictor(
            ffn_type="regression",
            input_dim=300,
            n_tasks=2,
            criterion=criterion,
        )
        assert isinstance(ffn, nn.RegressionFFN)
