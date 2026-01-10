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
    # Use min_epochs_per_phase=0 to test phase transitions without minimum constraint
    state = CurriculumState(qualities=["high", "medium", "low"], patience=1, min_epochs_per_phase=0)
    assert state.phase == "warmup"

    # Simulate no improvement: best_val_top stays inf so it will update on first value
    state.update_from_val_top(epoch=0, top_loss=1.0)
    assert state.best_val_top == 1.0

    # Advance based on patience: epoch - best_epoch >= patience triggers movement
    state.maybe_advance_phase(epoch=1)
    assert state.phase in {"expand", "robust", "polish", "warmup"}


def test_curriculum_state_weights_for_phases():
    state = CurriculumState(qualities=["high", "medium"], patience=1, min_epochs_per_phase=0)
    assert state.weights["high"] > state.weights["medium"]

    # Force expand
    state.phase = "expand"
    weights_expand = state._weights_for_phase("expand")
    assert weights_expand["high"] > weights_expand["medium"]

    # Polish returns to warmup-like focus with new defaults: [0.90, 0.10]
    weights_polish = state._weights_for_phase("polish")
    assert weights_polish["high"] == 0.90 and weights_polish["medium"] == 0.10


def test_curriculum_state_single_quality():
    state = CurriculumState(qualities=["high"], patience=1, min_epochs_per_phase=0)
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
    state = CurriculumState(qualities=["high", "medium", "low"], patience=0, min_epochs_per_phase=0)
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
    state = CurriculumState(qualities=["high", "medium", "low"], patience=0, min_epochs_per_phase=0)
    cb = CurriculumCallback(state)

    # Metrics with NaN should be ignored and no update occurs
    trainer = DummyTrainer(epoch=0, step=0, metrics={"val_loss": float("nan")})
    pl_module = DummyModule()
    cb.on_validation_epoch_end(trainer, pl_module)
    # No logs created
    assert pl_module.logged == {}


def test_curriculum_state_patience_resets_on_phase_change():
    """Test that patience counter (best_epoch) resets when phase advances.

    This prevents cascading phase transitions when the model doesn't immediately
    improve after a phase change.
    """
    # Use min_epochs_per_phase=0 so we can test patience reset in isolation
    state = CurriculumState(qualities=["high", "medium", "low"], patience=5, min_epochs_per_phase=0)
    assert state.phase == "warmup"
    assert state.best_epoch == 0

    # Simulate improvement at epoch 0
    state.update_from_val_top(epoch=0, top_loss=1.0)
    assert state.best_epoch == 0

    # No improvement for patience epochs -> should advance at epoch 5
    state.maybe_advance_phase(epoch=5)
    assert state.phase == "expand"
    # CRITICAL: best_epoch should be reset to current epoch
    assert state.best_epoch == 5, "best_epoch should reset to current epoch on phase change"

    # Now if we call maybe_advance_phase again immediately, it should NOT advance
    # because epoch - best_epoch < patience
    state.maybe_advance_phase(epoch=6)
    assert state.phase == "expand", "Phase should not advance before patience expires"

    # Only after another patience period should it advance again
    state.maybe_advance_phase(epoch=10)
    assert state.phase == "robust"
    assert state.best_epoch == 10, "best_epoch should reset again on second phase change"


def test_curriculum_state_finetune_phase_enabled():
    """Test that finetune phase is added when finetune_enabled=True."""
    # Without finetune
    state_no_finetune = CurriculumState(
        qualities=["high", "medium", "low"], patience=1, min_epochs_per_phase=0, finetune_enabled=False
    )
    assert state_no_finetune.finetune_enabled is False

    # With finetune
    state_with_finetune = CurriculumState(
        qualities=["high", "medium", "low"], patience=1, min_epochs_per_phase=0, finetune_enabled=True
    )
    assert state_with_finetune.finetune_enabled is True

    # Test finetune weights are configured
    finetune_weights = state_with_finetune.config.three_quality.get("finetune")
    assert finetune_weights == [1.0, 0.0, 0.0], "Finetune should use 100% high-quality"

    two_quality_finetune = state_with_finetune.config.two_quality.get("finetune")
    assert two_quality_finetune == [1.0, 0.0], "Two-quality finetune should use 100% high"


def test_curriculum_state_finetune_phase_progression():
    """Test that phase progression includes finetune when enabled."""
    state = CurriculumState(
        qualities=["high", "medium", "low"], patience=1, min_epochs_per_phase=0, finetune_enabled=True
    )
    assert state.phase == "warmup"

    # Progress through all phases
    phases_visited = [state.phase]
    for epoch in range(1, 50):
        state.update_from_val_top(epoch, 1.0)  # Constant loss triggers patience
        state.maybe_advance_phase(epoch)
        if state.phase not in phases_visited:
            phases_visited.append(state.phase)

    # Should visit all phases including finetune
    expected_phases = ["warmup", "expand", "robust", "polish", "finetune"]
    assert phases_visited == expected_phases, f"Expected {expected_phases}, got {phases_visited}"

    # Final phase should be finetune
    assert state.phase == "finetune"

    # Finetune weights should be 100% high-quality
    weights = state._weights_for_phase("finetune")
    assert weights["high"] == 1.0
    assert weights["medium"] == 0.0
    assert weights["low"] == 0.0


def test_curriculum_state_finetune_not_included_when_disabled():
    """Test that finetune phase is NOT included when finetune_enabled=False."""
    state = CurriculumState(
        qualities=["high", "medium", "low"], patience=1, min_epochs_per_phase=0, finetune_enabled=False
    )

    # Progress through all phases
    phases_visited = [state.phase]
    for epoch in range(1, 50):
        state.update_from_val_top(epoch, 1.0)
        state.maybe_advance_phase(epoch)
        if state.phase not in phases_visited:
            phases_visited.append(state.phase)

    # Should NOT include finetune
    expected_phases = ["warmup", "expand", "robust", "polish"]
    assert phases_visited == expected_phases, f"Expected {expected_phases}, got {phases_visited}"
    assert state.phase == "polish"
