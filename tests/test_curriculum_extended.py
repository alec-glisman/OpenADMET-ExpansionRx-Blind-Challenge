"""
Extended unit tests for admet.model.chemprop.curriculum module.

Tests curriculum state, phase transitions, and callback logging.
"""

from types import SimpleNamespace

import pytest

from admet.model.chemprop.curriculum import CurriculumCallback, CurriculumState


def _weights_equal(w1: dict, w2: dict, tol: float = 1e-6) -> bool:
    """Helper to compare weight dictionaries."""
    if set(w1.keys()) != set(w2.keys()):
        return False
    return all(abs(w1[k] - w2[k]) <= tol for k in w1)


class TestCurriculumState:
    """Tests for CurriculumState class."""

    def test_default_initialization(self) -> None:
        """Test default initialization with 3 qualities."""
        state = CurriculumState()
        assert state.qualities == ["high", "medium", "low"]
        assert state.phase == "warmup"
        assert state.patience == 3
        assert state.best_val_top == float("inf")

    def test_custom_qualities(self) -> None:
        """Test initialization with custom quality names."""
        state = CurriculumState(qualities=["excellent", "good", "fair", "poor"])
        assert len(state.qualities) == 4
        assert state.qualities[0] == "excellent"

    def test_empty_qualities_raises(self) -> None:
        """Test that empty qualities raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            CurriculumState(qualities=[])

    def test_target_metric_key(self) -> None:
        """Test that target_metric_key returns overall val_loss."""
        state = CurriculumState()
        assert state.target_metric_key() == "val_loss"

    def test_sampling_probs_sum_to_one(self) -> None:
        """Test that sampling probabilities sum to 1."""
        state = CurriculumState()
        probs = state.sampling_probs()
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_phase_progression_three_qualities(self) -> None:
        """Test full phase progression with 3 qualities."""
        state = CurriculumState(qualities=["high", "medium", "low"], patience=1)

        # Start in warmup
        assert state.phase == "warmup"

        # Simulate improvement at epoch 0, then stall
        state.update_from_val_top(0, 0.5)
        state.maybe_advance_phase(1)  # No improvement for 1 epoch
        assert state.phase == "expand"

        state.update_from_val_top(1, 0.45)
        state.maybe_advance_phase(2)
        assert state.phase == "robust"

        state.update_from_val_top(2, 0.4)
        state.maybe_advance_phase(3)
        assert state.phase == "polish"

        # Should stay at polish
        state.update_from_val_top(3, 0.35)
        state.maybe_advance_phase(4)
        assert state.phase == "polish"

    def test_no_advance_on_improvement(self) -> None:
        """Test that phase doesn't advance when loss improves."""
        state = CurriculumState(patience=2)

        state.update_from_val_top(0, 0.5)
        state.maybe_advance_phase(0)
        assert state.phase == "warmup"

        # Continuous improvement should keep us in warmup
        state.update_from_val_top(1, 0.4)
        state.maybe_advance_phase(1)
        assert state.phase == "warmup"

        state.update_from_val_top(2, 0.3)
        state.maybe_advance_phase(2)
        assert state.phase == "warmup"

    def test_weights_for_warmup_phase(self) -> None:
        """Test warmup phase weights (high quality focused)."""
        state = CurriculumState(qualities=["high", "medium", "low"])
        weights = state._weights_for_phase("warmup")
        assert weights["high"] == 0.9
        assert weights["medium"] == 0.1
        assert weights["low"] == 0.0

    def test_weights_for_expand_phase(self) -> None:
        """Test expand phase weights."""
        state = CurriculumState(qualities=["high", "medium", "low"])
        weights = state._weights_for_phase("expand")
        assert weights["high"] == 0.6
        assert weights["medium"] == 0.35
        assert weights["low"] == 0.05

    def test_weights_for_robust_phase(self) -> None:
        """Test robust phase weights."""
        state = CurriculumState(qualities=["high", "medium", "low"])
        weights = state._weights_for_phase("robust")
        assert weights["high"] == 0.4
        assert weights["medium"] == 0.4
        assert weights["low"] == 0.2

    def test_weights_for_polish_phase(self) -> None:
        """Test polish phase weights (back to high quality)."""
        state = CurriculumState(qualities=["high", "medium", "low"])
        weights = state._weights_for_phase("polish")
        assert weights["high"] == 1.0
        assert weights["medium"] == 0.0
        assert weights["low"] == 0.0


class TestCurriculumCallback:
    """Tests for CurriculumCallback class."""

    def test_callback_initialization(self) -> None:
        """Test callback initialization."""
        state = CurriculumState()
        callback = CurriculumCallback(state)
        assert callback.curr_state is state
        assert callback.monitor_metric is None
        assert callback._previous_phase == "warmup"

    def test_callback_with_custom_metric(self) -> None:
        """Test callback with custom monitor metric."""
        state = CurriculumState()
        callback = CurriculumCallback(state, monitor_metric="val_custom_loss")
        assert callback.monitor_metric == "val_custom_loss"

    def test_callback_updates_state(self, mocker) -> None:
        """Test that callback updates curriculum state on validation end."""
        state = CurriculumState(patience=1)
        callback = CurriculumCallback(state)

        # Create mock trainer and module
        trainer = SimpleNamespace(
            current_epoch=0,
            global_step=100,
            callback_metrics={"val_loss": 0.5},
        )
        pl_module = mocker.MagicMock()

        callback.on_validation_epoch_end(trainer, pl_module)
        assert state.best_val_top == 0.5

    def test_callback_handles_tensor_values(self, mocker) -> None:
        """Test that callback handles torch tensor values."""
        import torch

        state = CurriculumState()
        callback = CurriculumCallback(state)

        trainer = SimpleNamespace(
            current_epoch=0,
            global_step=100,
            callback_metrics={"val_loss": torch.tensor(0.42)},
        )
        pl_module = mocker.MagicMock()

        callback.on_validation_epoch_end(trainer, pl_module)
        assert state.best_val_top == pytest.approx(0.42)

    def test_callback_ignores_nan_values(self, mocker) -> None:
        """Test that callback ignores NaN validation loss."""
        state = CurriculumState()
        state.update_from_val_top(0, 0.5)  # Set initial value
        callback = CurriculumCallback(state)

        trainer = SimpleNamespace(
            current_epoch=1,
            global_step=200,
            callback_metrics={"val_loss": float("nan")},
        )
        pl_module = mocker.MagicMock()

        callback.on_validation_epoch_end(trainer, pl_module)
        # Should not update with NaN
        assert state.best_val_top == 0.5

    def test_callback_ignores_missing_metric(self, mocker) -> None:
        """Test that callback handles missing metric gracefully."""
        state = CurriculumState()
        callback = CurriculumCallback(state)

        trainer = SimpleNamespace(
            current_epoch=0,
            global_step=100,
            callback_metrics={},  # No val_loss
        )
        pl_module = mocker.MagicMock()

        # Should not raise
        callback.on_validation_epoch_end(trainer, pl_module)
        assert state.best_val_top == float("inf")

    def test_callback_logs_phase_transition(self, mocker) -> None:
        """Test that callback logs phase transitions."""
        state = CurriculumState(patience=1)
        callback = CurriculumCallback(state)

        # First epoch - establish baseline
        trainer0 = SimpleNamespace(
            current_epoch=0,
            global_step=100,
            callback_metrics={"val_loss": 0.5},
        )
        pl_module = mocker.MagicMock()
        callback.on_validation_epoch_end(trainer0, pl_module)

        # Second epoch - trigger phase change (no improvement)
        trainer1 = SimpleNamespace(
            current_epoch=1,
            global_step=200,
            callback_metrics={"val_loss": 0.55},  # Worse
        )

        # Patch the logging module at the stdlib level (curriculum imports it locally)
        mock_logger = mocker.MagicMock()
        mocker.patch("logging.getLogger", return_value=mock_logger)
        callback.on_validation_epoch_end(trainer1, pl_module)

        # Should have logged the phase transition
        assert state.phase == "expand"
        mock_logger.info.assert_called()

    def test_callback_logs_metrics_on_phase_change(self, mocker) -> None:
        """Test that callback logs metrics to pl_module on phase change."""
        state = CurriculumState(patience=1)
        callback = CurriculumCallback(state)

        pl_module = mocker.MagicMock()

        # First epoch
        trainer0 = SimpleNamespace(
            current_epoch=0,
            global_step=100,
            callback_metrics={"val_loss": 0.5},
        )
        callback.on_validation_epoch_end(trainer0, pl_module)

        # Reset mock
        pl_module.reset_mock()

        # Second epoch - trigger phase change
        trainer1 = SimpleNamespace(
            current_epoch=1,
            global_step=200,
            callback_metrics={"val_loss": 0.55},
        )
        callback.on_validation_epoch_end(trainer1, pl_module)

        # Should have logged curriculum_phase, curriculum_phase_epoch, and per-quality weights
        # With log_per_quality_metrics=True (default), we log: curriculum_phase,
        # curriculum_phase_epoch, curriculum_weight_high, curriculum_weight_medium, etc.
        assert pl_module.log.call_count >= 2
        logged_metrics = {call[0][0] for call in pl_module.log.call_args_list}
        assert "curriculum_phase" in logged_metrics
        assert "curriculum_phase_epoch" in logged_metrics


class TestCurriculumPhaseWeightsConsistency:
    """Tests for consistency of curriculum weights across phases."""

    @pytest.mark.parametrize(
        "n_qualities,expected_phases",
        [
            (1, ["warmup", "polish"]),
            (2, ["warmup", "expand", "polish"]),
            (3, ["warmup", "expand", "robust", "polish"]),
        ],
    )
    def test_phase_sequence_by_num_qualities(self, n_qualities: int, expected_phases: list[str]) -> None:
        """Test that phase sequence depends on number of qualities."""
        qualities = [f"q{i}" for i in range(n_qualities)]
        state = CurriculumState(qualities=qualities, patience=1)

        visited_phases = [state.phase]
        for epoch in range(1, len(expected_phases) + 5):
            state.update_from_val_top(epoch - 1, 0.5)  # No improvement
            state.maybe_advance_phase(epoch)
            if state.phase not in visited_phases:
                visited_phases.append(state.phase)

        assert visited_phases == expected_phases

    def test_weights_always_sum_to_one(self) -> None:
        """Test that weights always sum to 1 across all phases."""
        for n_qualities in [1, 2, 3, 4, 5]:
            qualities = [f"q{i}" for i in range(n_qualities)]
            state = CurriculumState(qualities=qualities)

            for phase in ["warmup", "expand", "robust", "polish"]:
                weights = state._weights_for_phase(phase)
                total = sum(weights.values())
                assert total == pytest.approx(1.0), f"Weights don't sum to 1 for {phase} with {n_qualities} qualities"

    def test_weights_non_negative(self) -> None:
        """Test that all weights are non-negative."""
        state = CurriculumState(qualities=["high", "medium", "low"])

        for phase in ["warmup", "expand", "robust", "polish"]:
            weights = state._weights_for_phase(phase)
            for q, w in weights.items():
                assert w >= 0, f"Negative weight {w} for {q} in {phase}"
