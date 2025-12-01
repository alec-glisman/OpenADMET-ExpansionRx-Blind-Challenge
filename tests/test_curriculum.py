from admet.model.chemprop.curriculum import CurriculumState


def _weights_equal(w1, w2, tol: float = 1e-6):
    if set(w1.keys()) != set(w2.keys()):
        return False
    for k in w1:
        if abs(w1[k] - w2[k]) > tol:
            return False
    return True


def test_curriculum_n1_phase_progression_weights():
    # Single quality: only 'warmup' and 'polish' proceed, weights are [1.0]
    cs = CurriculumState(qualities=["high"], patience=1)
    assert cs.phase == "warmup"
    assert _weights_equal(cs.weights, {"high": 1.0})
    # simulate improvement at epoch 0 followed by epoch 1 advancing
    cs.update_from_val_top(0, 0.5)
    cs.maybe_advance_phase(1)
    assert cs.phase == "polish"
    assert _weights_equal(cs.weights, {"high": 1.0})


def test_curriculum_n2_phase_progression_weights():
    cs = CurriculumState(qualities=["high", "medium"], patience=1)
    assert cs.phase == "warmup"
    assert _weights_equal(cs.weights, {"high": 0.9, "medium": 0.1})
    # warmup -> expand
    cs.update_from_val_top(0, 0.5)
    cs.maybe_advance_phase(1)
    assert cs.phase == "expand"
    assert _weights_equal(cs.weights, {"high": 0.6, "medium": 0.4})
    # expand -> polish
    cs.update_from_val_top(1, 0.4)
    cs.maybe_advance_phase(2)
    assert cs.phase == "polish"
    assert _weights_equal(cs.weights, {"high": 1.0, "medium": 0.0})


def test_curriculum_n3_phase_progression_weights():
    cs = CurriculumState(qualities=["high", "medium", "low"], patience=1)
    assert cs.phase == "warmup"
    assert _weights_equal(cs.weights, {"high": 0.9, "medium": 0.1, "low": 0.0})
    # warmup -> expand
    cs.update_from_val_top(0, 0.5)
    cs.maybe_advance_phase(1)
    assert cs.phase == "expand"
    assert _weights_equal(cs.weights, {"high": 0.6, "medium": 0.35, "low": 0.05})
    # expand -> robust
    cs.update_from_val_top(1, 0.4)
    cs.maybe_advance_phase(2)
    assert cs.phase == "robust"
    assert _weights_equal(cs.weights, {"high": 0.4, "medium": 0.4, "low": 0.2})
    # robust -> polish
    cs.update_from_val_top(2, 0.3)
    cs.maybe_advance_phase(3)
    assert cs.phase == "polish"
    assert _weights_equal(cs.weights, {"high": 1.0, "medium": 0.0, "low": 0.0})
