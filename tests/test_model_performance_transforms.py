import numpy as np
from admet.visualize.model_performance import to_linear
import pytest


def test_to_linear_identity_and_transform():
    arr = np.array([[-1.0], [0.0], [1.0]])
    # Non-LogD endpoint: should apply 10^x
    out = to_linear(arr.flatten(), "KSOL")
    if not np.allclose(out, np.power(10.0, arr.flatten())):
        pytest.fail("to_linear did not apply 10^x transform as expected for non-LogD endpoint")
    # LogD identity
    out2 = to_linear(arr.flatten(), "LogD")
    if not np.allclose(out2, arr.flatten()):
        pytest.fail("to_linear did not return identity for LogD endpoint")
