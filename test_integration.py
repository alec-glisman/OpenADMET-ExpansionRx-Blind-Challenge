"""Test task affinity integration with ChempropModel."""

import logging

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from admet.model.chemprop.config import ChempropConfig, TaskAffinityConfig
from admet.model.chemprop.model import ChempropModel

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create sample data
smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "C1=CC=C(C=C1)CC(C(=O)O)N",  # Phenylalanine
    "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
] * 4  # Repeat to have enough samples

np.random.seed(42)
df = pd.DataFrame(
    {
        "SMILES": smiles,
        "LogD": np.random.randn(len(smiles)),
        "KSOL": np.random.randn(len(smiles)),
        "CLint": np.random.randn(len(smiles)),
    }
)

# Split data
train_size = int(0.8 * len(df))
df_train = df.iloc[:train_size].copy()
df_val = df.iloc[train_size:].copy()

# Test 1: Model without task affinity
print("\n" + "=" * 70)
print("Test 1: ChempropModel without task affinity")
print("=" * 70)

model1 = ChempropModel(
    df_train=df_train,
    df_validation=df_val,
    smiles_col="SMILES",
    target_cols=["LogD", "KSOL", "CLint"],
    mlflow_tracking=False,
    progress_bar=False,
)

# Check that affinity is None
assert model1.task_affinity_config is None
assert model1.task_affinity_matrix is None
assert model1.task_groups is None
print("✓ Model initialized without task affinity")

# Test 2: Model with task affinity enabled
print("\n" + "=" * 70)
print("Test 2: ChempropModel with task affinity enabled")
print("=" * 70)

task_affinity_config = TaskAffinityConfig(
    enabled=True,
    affinity_epochs=1,
    affinity_batch_size=4,
    n_groups=2,
    clustering_method="agglomerative",
)

model2 = ChempropModel(
    df_train=df_train,
    df_validation=df_val,
    smiles_col="SMILES",
    target_cols=["LogD", "KSOL", "CLint"],
    task_affinity_config=task_affinity_config,
    mlflow_tracking=False,
    progress_bar=False,
)

# Check that config is stored
assert model2.task_affinity_config is not None
assert model2.task_affinity_config.enabled is True
print("✓ Task affinity config stored")

# Compute task affinity manually
model2._compute_task_affinity()

# Check that affinity results are stored
assert model2.task_affinity_matrix is not None
assert model2.task_groups is not None
assert model2.task_group_indices is not None
print("✓ Task affinity computed and stored")
print(f"  Affinity matrix shape: {model2.task_affinity_matrix.shape}")
print(f"  Task groups: {model2.task_groups}")
print(f"  Task group indices: {model2.task_group_indices}")

# Test 3: Model from config
print("\n" + "=" * 70)
print("Test 3: ChempropModel from config with task affinity")
print("=" * 70)

config_dict = {
    "data": {
        "data_dir": ".",
        "smiles_col": "SMILES",
        "target_cols": ["LogD", "KSOL", "CLint"],
    },
    "model": {
        "ffn_type": "branched",
        "depth": 3,
        "hidden_dim": 300,
    },
    "optimization": {
        "max_epochs": 2,
        "batch_size": 4,
    },
    "mlflow": {
        "tracking": False,
    },
    "task_affinity": {
        "enabled": True,
        "affinity_epochs": 1,
        "affinity_batch_size": 4,
        "n_groups": 2,
    },
}

config = OmegaConf.merge(
    OmegaConf.structured(ChempropConfig),
    OmegaConf.create(config_dict),
)

model3 = ChempropModel.from_config(
    config,
    df_train=df_train,
    df_validation=df_val,
)

assert model3.task_affinity_config is not None
assert model3.task_affinity_config.enabled is True
print("✓ Model created from config with task affinity")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
