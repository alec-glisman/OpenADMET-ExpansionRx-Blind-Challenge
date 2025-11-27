"""Tests for model performance transformation helpers (log <-> linear)."""

from __future__ import annotations

import numpy as np
import pytest

from admet.visualize.model_performance import to_linear


@pytest.mark.unit
def test_to_linear_identity_and_transform() -> None:
    arr = np.array([[-1.0], [0.0], [1.0]])
    out = to_linear(arr.flatten(), "KSOL")
    assert np.allclose(out, np.power(10.0, arr.flatten()))
    out2 = to_linear(arr.flatten(), "LogD")
    assert np.allclose(out2, arr.flatten())
