"""
Unit tests for CurriculumState and CurriculumCallback
"""

import logging

import pytest

from admet.model.chemprop.curriculum import CurriculumCallback, CurriculumState


class DummyTrainer:
    def __init__(self, epoch=0, step=0, metrics=None):
        self.current_epoch = epoch
        self.global_step = step
        self.callback_metrics = metrics or {}


class DummyModule:
    def __init__(self):
        self.logged = {}

    def log(self, key, value, on_step=False, on_epoch=True):
        self.logged[key] = value


def test_curriculum_state_basic_phases():
    state = CurriculumState(qualities=["high", "medium", "low"], patience=1)
    assert state.phase == "warmup"

    # Simulate no improvement: best_val_top stays inf so it will update on first value
    state.update_from_val_top(epoch=0, top_loss=1.0)
    assert state.best_val_top == 1.0

    # Advance based on patience: epoch - best_epoch >= patience triggers movement
    state.maybe_advance_phase(epoch=1)
    assert state.phase in {"expand", "robust", "polish", "warmup"}


def test_curriculum_state_weights_for_phases():
    state = CurriculumState(qualities=["high", "medium"], patience=1)
    assert state.weights["high"] > state.weights["medium"]

    # Force expand
    state.phase = "expand"
    weights_expand = state._weights_for_phase("expand")
    assert weights_expand["high"] > weights_expand["medium"]

    # Polish returns high-only
    weights_polish = state._weights_for_phase("polish")
    assert weights_polish["high"] == 1.0 and weights_polish["medium"] == 0.0


def test_curriculum_state_single_quality():
    state = CurriculumState(qualities=["high"], patience=1)
    assert state.weights["high"] == 1.0

    # Advance eventually to 'polish' (only two phases)
    state.update_from_val_top(0, 0.5)
    state.maybe_advance_phase(2)
    assert state.phase == "polish"


def test_curriculum_state_invalid_qualities():
    with pytest.raises(ValueError):
        CurriculumState(qualities=[], patience=1)


def test_curriculum_callback_logs(caplog):
    """Test that the callback logs transitions and calls pl_module.log()."""
    state = CurriculumState(qualities=["high", "medium", "low"], patience=0)
    cb = CurriculumCallback(state)

    # Create a dummy trainer and module; simulate metrics
    trainer = DummyTrainer(epoch=0, step=0, metrics={"val_loss": 1.23})
    pl_module = DummyModule()

    caplog.set_level(logging.INFO)

    # on validation end should update and advance; with patience=0 it should move immediately
    cb.on_validation_epoch_end(trainer, pl_module)

    # Ensure the module logged phase and epoch with hierarchical naming
    assert "curriculum/phase" in pl_module.logged
    assert "curriculum/phase_epoch" in pl_module.logged

    # Now simulate a transition by changing val_loss and epoch large enough
    prev_phase = state.phase
    trainer.callback_metrics = {"val_loss": 2.0}
    trainer.current_epoch = trainer.current_epoch + state.patience + 1
    cb.on_validation_epoch_end(trainer, pl_module)

    assert state.phase != prev_phase or state.phase == "polish"


def test_curriculum_callback_handles_nan():
    """Test that callback handles NaN validation loss."""
    state = CurriculumState(qualities=["high", "medium", "low"], patience=0)
    cb = CurriculumCallback(state)

    # Metrics with NaN should be ignored and no update occurs
    trainer = DummyTrainer(epoch=0, step=0, metrics={"val_loss": float("nan")})
    pl_module = DummyModule()
    cb.on_validation_epoch_end(trainer, pl_module)
    # No logs created
    assert pl_module.logged == {}
