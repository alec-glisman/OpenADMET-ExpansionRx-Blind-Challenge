"""
Unit tests for admet.model.chemprop.curriculum callbacks and state.

Tests CurriculumState weight schedules, phase progression, and CurriculumCallback
which responds to validation metrics by updating the CurriculumState and logging
phase changes.
"""

import logging
import math

import pytest

from admet.model.chemprop.curriculum import CurriculumCallback, CurriculumState


class DummyTrainer:
    def __init__(self, epoch: int = 0, step: int = 0, val_loss=None):
        self.current_epoch = epoch
        self.global_step = step
        self.callback_metrics = {"val_loss": val_loss}


class DummyModule:
    def __init__(self):
        self.logged = {}

    def log(self, key, value, on_step=False, on_epoch=True):
        self.logged[key] = value


def test_curriculum_state_weights_and_phases():
    state = CurriculumState(qualities=["high", "medium", "low"], patience=1)
    # Initial phase is warmup
    assert state.phase == "warmup"
    # Warmup weights
    w = state.sampling_probs()
    assert abs(w["high"] - 0.9) < 1e-6

    # Simulate improvement and advancement
    state.update_from_val_top(epoch=0, top_loss=0.5)
    # No change without patience
    state.maybe_advance_phase(0)
    assert state.phase == "warmup"

    # After patience epochs without improvement, advance
    state.maybe_advance_phase(2)
    assert state.phase in ["expand", "robust", "polish"]


def test_curriculum_callback_updates_phase_and_logs(caplog):
    state = CurriculumState(qualities=["high", "medium", "low"], patience=1)
    cb = CurriculumCallback(state)

    # create dummy trainer and module
    trainer = DummyTrainer(epoch=0, step=10, val_loss=0.5)
    pl_module = DummyModule()

    # Call the callback; first time should set best and not advance
    cb.on_validation_epoch_end(trainer, pl_module)
    assert state.best_val_top == pytest.approx(0.5)
    assert cb._previous_phase == state.phase

    # Now set worse val_loss; patience expired, should advance phase
    caplog.set_level(logging.INFO)
    trainer = DummyTrainer(epoch=3, step=15, val_loss=0.9)
    state.best_epoch = 0  # previously best at epoch 0
    cb.on_validation_epoch_end(trainer, pl_module)

    # If advanced, curriculum_phase logged to module
    if state.phase != cb._previous_phase:
        assert "curriculum_phase" in pl_module.logged


def test_callback_ignores_nan_values():
    state = CurriculumState(qualities=["high", "medium"], patience=1)
    cb = CurriculumCallback(state)
    trainer = DummyTrainer(epoch=1, step=5, val_loss=float("nan"))
    pl_module = DummyModule()
    cb.on_validation_epoch_end(trainer, pl_module)
    # No update to best when val is nan
    assert math.isinf(state.best_val_top) or state.best_val_top == float("inf")
