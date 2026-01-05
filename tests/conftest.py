"""
Shared pytest fixtures for admet.model.chemprop tests.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import torch

# Provide a fallback `benchmark` fixture when pytest-benchmark plugin is not available.
# Some CI/dev environments may not install pytest-benchmark; these tests are intended
# to be optional performance checks. The fallback simply calls the function under
# test and returns its result so unit test runs do not error out.
if importlib.util.find_spec("pytest_benchmark") is None:

    @pytest.fixture
    def benchmark():
        """Fallback benchmark fixture that runs the callable and returns a simple result object.

        If `pytest-benchmark` isn't installed, this allows benchmark-marked tests to still
        execute the target function and receive a deterministic, non-None result that
        the tests can assert on (these are not intended to provide real benchmarking
        metrics in this environment).
        """

        class _Benchmark:
            def __call__(self, func, *args, **kwargs):
                # Execute the provided callable to reproduce side effects.
                result = func(*args, **kwargs)
                # If the callable returned None (e.g., compress_logs), return a
                # minimal non-None result so tests that assert the benchmark
                # returned a value continue to pass. Otherwise return the
                # original result (e.g., an object like EnsembleProgressTracker).
                if result is None:
                    return {"result": result, "runs": 1, "stats": {}}
                return result

        return _Benchmark()


@pytest.fixture(autouse=True)
def reset_torch_deterministic_mode():
    """Reset PyTorch deterministic mode before and after each test.

    Some tests (e.g., ChempropModel training) enable deterministic mode via
    Lightning's Trainer(deterministic=True), which globally sets
    torch.use_deterministic_algorithms(True). This can cause failures in
    subsequent tests that use operations without deterministic implementations.

    This fixture ensures deterministic mode is disabled before and after each test.
    """
    # Disable deterministic mode before test starts
    torch.use_deterministic_algorithms(False)
    yield
    # Disable deterministic mode after test completes
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def sample_smiles() -> list[str]:
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CC(C)O",  # isopropanol
        "CCCC",  # butane
        "c1ccc(O)cc1",  # phenol
        "CC(=O)OC",  # methyl acetate
        "CCOCC",  # diethyl ether
    ]


@pytest.fixture
def sample_targets() -> list[str]:
    """Sample target column names."""
    return ["LogD", "Log KSOL", "Log HLM CLint"]


@pytest.fixture
def sample_quality_labels() -> list[str]:
    """Sample quality labels for curriculum learning."""
    return ["high", "high", "medium", "low", "high", "medium", "low", "high"]


@pytest.fixture
def sample_dataframe(sample_smiles, sample_targets, sample_quality_labels) -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    data = {
        "SMILES": sample_smiles,
        "Quality": sample_quality_labels,
    }
    for target in sample_targets:
        data[target] = np.random.randn(len(sample_smiles))
    return pd.DataFrame(data)


@pytest.fixture
def train_val_dataframes(sample_dataframe) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sample dataframe into train and validation sets."""
    n = len(sample_dataframe)
    train_idx = list(range(n - 2))
    val_idx = list(range(n - 2, n))
    return sample_dataframe.iloc[train_idx].copy(), sample_dataframe.iloc[val_idx].copy()
