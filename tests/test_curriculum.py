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
    # New conservative defaults: [0.85, 0.15]
    assert _weights_equal(cs.weights, {"high": 0.85, "medium": 0.15})
    # warmup -> expand
    cs.update_from_val_top(0, 0.5)
    cs.maybe_advance_phase(1)
    assert cs.phase == "expand"
    # New conservative defaults: [0.65, 0.35]
    assert _weights_equal(cs.weights, {"high": 0.65, "medium": 0.35})
    # expand -> polish
    cs.update_from_val_top(1, 0.4)
    cs.maybe_advance_phase(2)
    assert cs.phase == "polish"
    # New conservative defaults: [0.75, 0.25] - maintains diversity
    assert _weights_equal(cs.weights, {"high": 0.75, "medium": 0.25})


def test_curriculum_n3_phase_progression_weights():
    cs = CurriculumState(qualities=["high", "medium", "low"], patience=1)
    assert cs.phase == "warmup"
    # New conservative defaults: [0.80, 0.15, 0.05]
    assert _weights_equal(cs.weights, {"high": 0.80, "medium": 0.15, "low": 0.05})
    # warmup -> expand
    cs.update_from_val_top(0, 0.5)
    cs.maybe_advance_phase(1)
    assert cs.phase == "expand"
    # New conservative defaults: [0.60, 0.30, 0.10]
    assert _weights_equal(cs.weights, {"high": 0.60, "medium": 0.30, "low": 0.10})
    # expand -> robust
    cs.update_from_val_top(1, 0.4)
    cs.maybe_advance_phase(2)
    assert cs.phase == "robust"
    # New conservative defaults: [0.50, 0.35, 0.15]
    assert _weights_equal(cs.weights, {"high": 0.50, "medium": 0.35, "low": 0.15})
    # robust -> polish
    cs.update_from_val_top(2, 0.3)
    cs.maybe_advance_phase(3)
    assert cs.phase == "polish"
    # New conservative defaults: [0.70, 0.20, 0.10] - maintains diversity
    assert _weights_equal(cs.weights, {"high": 0.70, "medium": 0.20, "low": 0.10})
