import numpy as np
from admet.visualize.model_performance import to_linear


def test_to_linear_identity_and_transform():
    arr = np.array([[-1.0], [0.0], [1.0]])
    # Non-LogD endpoint: should apply 10^x
    out = to_linear(arr.flatten(), "KSOL")
    assert np.allclose(out, np.power(10.0, arr.flatten()))
    # LogD identity
    out2 = to_linear(arr.flatten(), "LogD")
    assert np.allclose(out2, arr.flatten())
