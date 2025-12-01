"""
Unit tests for admet.model.chemprop.ffn module.

Tests custom FFN architectures: BranchedFFN and MixtureOfExpertsRegressionFFN.
"""

import pytest
import torch

from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN


class TestMixtureOfExpertsRegressionFFN:
    """Tests for MixtureOfExpertsRegressionFFN class."""

    @pytest.fixture
    def moe_ffn(self) -> MixtureOfExpertsRegressionFFN:
        """Create a basic MoE FFN for testing."""
        return MixtureOfExpertsRegressionFFN(
            n_tasks=3,
            n_experts=4,
            input_dim=128,
            hidden_dim=64,
            n_layers=2,
            dropout=0.1,
        )

    def test_initialization(self, moe_ffn: MixtureOfExpertsRegressionFFN) -> None:
        """Test MoE FFN initialization."""
        assert moe_ffn.n_tasks == 3
        assert moe_ffn.n_experts == 4
        assert moe_ffn.input_dim == 128

    def test_forward_shape(self, moe_ffn: MixtureOfExpertsRegressionFFN) -> None:
        """Test forward pass output shape."""
        batch_size = 16
        x = torch.randn(batch_size, 128)
        output = moe_ffn(x)
        assert output.shape == (batch_size, 3)

    def test_forward_different_batch_sizes(self, moe_ffn: MixtureOfExpertsRegressionFFN) -> None:
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 128)
            output = moe_ffn(x)
            assert output.shape == (batch_size, 3)

    def test_expert_count_affects_gates(self) -> None:
        """Test that number of experts affects gating network."""
        for n_experts in [2, 4, 8]:
            ffn = MixtureOfExpertsRegressionFFN(
                n_tasks=2,
                n_experts=n_experts,
                input_dim=64,
                hidden_dim=32,
            )
            x = torch.randn(8, 64)
            output = ffn(x)
            assert output.shape == (8, 2)

    def test_gradient_flow(self, moe_ffn: MixtureOfExpertsRegressionFFN) -> None:
        """Test that gradients flow through the network."""
        x = torch.randn(8, 128, requires_grad=True)
        output = moe_ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestBranchedFFN:
    """Tests for BranchedFFN class."""

    @pytest.fixture
    def branched_ffn(self) -> BranchedFFN:
        """Create a basic branched FFN for testing."""
        return BranchedFFN(
            task_groups=[[0], [1], [2]],  # One task per branch
            n_tasks=3,
            input_dim=128,
            hidden_dim=64,
            trunk_n_layers=2,
            trunk_hidden_dim=96,
            trunk_dropout=0.1,
        )

    def test_initialization(self, branched_ffn: BranchedFFN) -> None:
        """Test branched FFN initialization."""
        assert branched_ffn.n_tasks == 3
        assert len(branched_ffn.task_groups) == 3

    def test_forward_shape(self, branched_ffn: BranchedFFN) -> None:
        """Test forward pass output shape."""
        batch_size = 16
        x = torch.randn(batch_size, 128)
        output = branched_ffn(x)
        assert output.shape == (batch_size, 3)

    def test_grouped_tasks(self) -> None:
        """Test branched FFN with grouped tasks."""
        # Group tasks: [0,1] share branch, [2] separate
        ffn = BranchedFFN(
            task_groups=[[0, 1], [2]],
            n_tasks=3,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=1,
            trunk_hidden_dim=48,
        )
        x = torch.randn(8, 64)
        output = ffn(x)
        assert output.shape == (8, 3)

    def test_single_branch_all_tasks(self) -> None:
        """Test branched FFN with all tasks in one branch."""
        ffn = BranchedFFN(
            task_groups=[[0, 1, 2, 3]],
            n_tasks=4,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=1,
            trunk_hidden_dim=48,
        )
        x = torch.randn(8, 64)
        output = ffn(x)
        assert output.shape == (8, 4)

    def test_gradient_flow(self, branched_ffn: BranchedFFN) -> None:
        """Test that gradients flow through all branches."""
        x = torch.randn(8, 128, requires_grad=True)
        output = branched_ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_batch_sizes(self, branched_ffn: BranchedFFN) -> None:
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 128)
            output = branched_ffn(x)
            assert output.shape == (batch_size, 3)


class TestFFNWithTaskWeights:
    """Tests for FFN architectures with task weights."""

    def test_moe_with_task_weights(self) -> None:
        """Test MoE FFN with task weights."""
        task_weights = torch.tensor([1.0, 2.0, 0.5])
        ffn = MixtureOfExpertsRegressionFFN(
            n_tasks=3,
            n_experts=4,
            input_dim=64,
            hidden_dim=32,
            task_weights=task_weights,
        )
        x = torch.randn(8, 64)
        output = ffn(x)
        assert output.shape == (8, 3)

    def test_branched_with_task_weights(self) -> None:
        """Test branched FFN with task weights."""
        task_weights = torch.tensor([1.0, 2.0, 0.5])
        ffn = BranchedFFN(
            task_groups=[[0], [1], [2]],
            n_tasks=3,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=1,
            trunk_hidden_dim=48,
            task_weights=task_weights,
        )
        x = torch.randn(8, 64)
        output = ffn(x)
        assert output.shape == (8, 3)


class TestFFNDropout:
    """Tests for dropout in FFN architectures."""

    def test_moe_dropout_training_mode(self) -> None:
        """Test MoE dropout is active in training mode."""
        ffn = MixtureOfExpertsRegressionFFN(
            n_tasks=2,
            n_experts=2,
            input_dim=64,
            hidden_dim=32,
            dropout=0.5,  # High dropout
        )
        ffn.train()

        x = torch.randn(32, 64)
        outputs = [ffn(x) for _ in range(5)]

        # Outputs should vary due to dropout
        # Check that not all outputs are identical
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause output variation"

    def test_moe_dropout_eval_mode(self) -> None:
        """Test MoE dropout is disabled in eval mode."""
        ffn = MixtureOfExpertsRegressionFFN(
            n_tasks=2,
            n_experts=2,
            input_dim=64,
            hidden_dim=32,
            dropout=0.5,
        )
        ffn.eval()

        x = torch.randn(32, 64)
        with torch.no_grad():
            outputs = [ffn(x) for _ in range(5)]

        # All outputs should be identical in eval mode
        for output in outputs[1:]:
            assert torch.allclose(outputs[0], output)


class TestMoEGateNetwork:
    """Additional tests for MoE gating network."""

    def test_gate_custom_dimensions(self) -> None:
        """Test MoE with custom gate network dimensions."""
        ffn = MixtureOfExpertsRegressionFFN(
            n_experts=3,
            n_tasks=2,
            input_dim=64,
            hidden_dim=32,
            gate_hidden_dim=48,
            gate_n_layers=2,
        )
        x = torch.randn(8, 64)
        output = ffn(x)
        assert output.shape == (8, 2)

    def test_encode_method(self) -> None:
        """Test encoding at different depths."""
        ffn = MixtureOfExpertsRegressionFFN(
            n_experts=2,
            n_tasks=2,
            input_dim=64,
            hidden_dim=32,
            n_layers=2,
        )
        x = torch.randn(8, 64)
        # encode uses first expert's layers
        encoded = ffn.encode(x, 1)
        assert encoded is not None


class TestBranchedFFNValidation:
    """Tests for BranchedFFN input validation."""

    def test_invalid_task_groups_gaps(self) -> None:
        """Test that invalid task groups with gaps raise error."""
        with pytest.raises(ValueError, match="partition"):
            BranchedFFN(
                task_groups=[[0], [2]],  # Missing task 1
                n_tasks=3,
                input_dim=64,
                hidden_dim=32,
                trunk_n_layers=1,
                trunk_hidden_dim=48,
            )

    def test_invalid_task_groups_duplicates(self) -> None:
        """Test that duplicate task indices raise error."""
        with pytest.raises(ValueError, match="partition"):
            BranchedFFN(
                task_groups=[[0, 1], [1, 2]],  # Task 1 duplicated
                n_tasks=3,
                input_dim=64,
                hidden_dim=32,
                trunk_n_layers=1,
                trunk_hidden_dim=48,
            )


class TestBranchedFFNEncode:
    """Tests for BranchedFFN encode method."""

    def test_encode_zero_depth(self) -> None:
        """Test encode with depth 0 returns input."""
        ffn = BranchedFFN(
            task_groups=[[0], [1]],
            n_tasks=2,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=1,
            trunk_hidden_dim=48,
        )
        x = torch.randn(8, 64)
        encoded = ffn.encode(x, 0)
        assert torch.equal(encoded, x)

    def test_encode_trunk_depth(self) -> None:
        """Test encode within trunk depth."""
        ffn = BranchedFFN(
            task_groups=[[0], [1]],
            n_tasks=2,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=2,
            trunk_hidden_dim=48,
        )
        x = torch.randn(8, 64)
        encoded = ffn.encode(x, 1)
        assert encoded.shape[0] == 8

    def test_encode_beyond_trunk(self) -> None:
        """Test encode beyond trunk (into branches)."""
        ffn = BranchedFFN(
            task_groups=[[0], [1]],
            n_tasks=2,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=1,
            trunk_hidden_dim=48,
            n_layers=2,
        )
        x = torch.randn(8, 64)
        encoded = ffn.encode(x, 2)  # 1 trunk + 1 branch layer
        assert encoded.shape[0] == 8


class TestFFNHyperparameters:
    """Tests for FFN hyperparameter saving."""

    def test_moe_hparams_saved(self) -> None:
        """Test MoE hyperparameters are saved correctly."""
        ffn = MixtureOfExpertsRegressionFFN(
            n_experts=3,
            n_tasks=2,
            input_dim=64,
            hidden_dim=32,
            n_layers=2,
            dropout=0.1,
        )
        assert ffn.hparams["n_experts"] == 3
        assert ffn.hparams["n_tasks"] == 2
        assert ffn.hparams["input_dim"] == 64

    def test_branched_hparams_saved(self) -> None:
        """Test BranchedFFN hyperparameters are saved correctly."""
        ffn = BranchedFFN(
            task_groups=[[0, 1], [2]],
            n_tasks=3,
            input_dim=64,
            hidden_dim=32,
            trunk_n_layers=2,
            trunk_hidden_dim=48,
        )
        assert ffn.hparams["task_groups"] == [[0, 1], [2]]
        assert ffn.hparams["n_tasks"] == 3
        assert ffn.hparams["trunk_n_layers"] == 2
