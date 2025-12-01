"""
Shared pytest fixtures for admet.model.chemprop tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
