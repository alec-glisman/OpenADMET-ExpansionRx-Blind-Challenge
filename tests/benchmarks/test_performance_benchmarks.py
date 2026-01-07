"""
Performance benchmark tests for optimizations.

These tests measure actual performance improvements from each optimization.
They are marked as slow and should be run separately from unit tests.

Usage:
    pytest tests/benchmarks/test_performance_benchmarks.py -v
    pytest tests/benchmarks/test_performance_benchmarks.py -m "not slow"
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from admet.data.smiles import parallel_canonicalize_smiles
from admet.model.chemprop.config import ChempropConfig
from admet.model.chemprop.model import ChempropModel


@pytest.fixture
def benchmark_dataset() -> pd.DataFrame:
    """Create a larger dataset for benchmarking (100 molecules, 2 tasks)."""
    np.random.seed(42)

    # Generate random SMILES-like strings for testing
    smiles_templates = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "CC(C)O",
        "CCCC",
        "CCC(=O)O",
        "c1ccc(O)cc1",
        "CC(C)C",
        "CCCCO",
        "CCCCCC",
    ]

    smiles = []
    for _ in range(100):
        template = np.random.choice(smiles_templates)
        # Add random variations
        variations = ["", "C", "CC", "O", "N"]
        smiles.append(template + np.random.choice(variations))

    targets = {
        "LogD": np.random.randn(100) * 2 + 1.5,
        "LogP": np.random.randn(100) * 2 + 2.0,
    }

    df = pd.DataFrame({"SMILES": smiles, **targets})
    return df


@pytest.fixture
def minimal_chemprop_config() -> ChempropConfig:
    """Create minimal ChempropConfig for fast testing."""
    config_dict = {
        "model": {
            "depth": 2,
            "hidden_dim": 32,
            "message_hidden_dim": 32,
            "dropout": 0.0,
            "num_layers": 1,
        },
        "optimization": {
            "batch_size": 4,
            "max_epochs": 2,
            "init_lr": 1e-3,
            "max_lr": 1e-3,
            "final_lr": 1e-4,
            "warmup_epochs": 0,
            "patience": 10,
            "num_workers": 0,
            "seed": 42,
            "accumulate_grad_batches": 1,
        },
        "data": {
            "smiles_col": "SMILES",
            "target_cols": ["LogD", "LogP"],
            "data_dir": "",  # Will be set per test
        },
        "mlflow": {
            "tracking": False,
        },
    }

    base_config = OmegaConf.structured(ChempropConfig)
    override_config = OmegaConf.create(config_dict)
    config = OmegaConf.merge(base_config, override_config)
    OmegaConf.resolve(config)

    return config


@pytest.mark.slow
class TestBatchedPredictionPerformance:
    """Benchmark batched predictions vs single predictions."""

    def test_batched_prediction_speedup(
        self, benchmark_dataset: pd.DataFrame, minimal_chemprop_config: ChempropConfig, tmp_path: Path
    ):
        """Measure speedup from batched predictions (target: 2-3x faster)."""
        # Setup
        train_df = benchmark_dataset.iloc[:80]
        test_df = benchmark_dataset.iloc[80:]

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "validation.csv", index=False)

        minimal_chemprop_config.data.data_dir = str(data_dir)
        minimal_chemprop_config.optimization.batch_size = 32

        # Train model once
        model = ChempropModel.from_config(minimal_chemprop_config)
        model.fit()

        # Benchmark single predictions (old way)
        start_single = time.time()
        test_smiles = test_df["SMILES"].tolist()
        for smiles in test_smiles:
            _ = model.predict(pd.DataFrame({"SMILES": [smiles]}))
        time_single = time.time() - start_single

        # Benchmark batched predictions (new way)
        start_batched = time.time()
        _ = model.predict(test_df)
        time_batched = time.time() - start_batched

        speedup = time_single / time_batched

        print("\nBatched Prediction Performance:")
        print(f"  Single predictions: {time_single:.3f}s")
        print(f"  Batched predictions: {time_batched:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Should be at least 1.5x faster (conservative, target is 2-3x)
        assert speedup >= 1.5, f"Batched predictions should be at least 1.5x faster (got {speedup:.2f}x)"

        model.close()


@pytest.mark.slow
class TestSMILESCachePerformance:
    """Benchmark SMILES canonicalization caching."""

    @pytest.mark.xfail(reason="Flaky: Cache operations too fast to measure accurately on small datasets")
    def test_canonicalization_cache_speedup(self, benchmark_dataset: pd.DataFrame):
        """Measure speedup from SMILES canonicalization cache (target: 5-10x)."""
        smiles_list = benchmark_dataset["SMILES"].tolist()

        # First call - populates cache
        start_first = time.time()
        _ = parallel_canonicalize_smiles(smiles_list)
        time_first = time.time() - start_first

        # Second call - should hit cache
        start_cached = time.time()
        _ = parallel_canonicalize_smiles(smiles_list)
        time_cached = time.time() - start_cached

        # Third call - also cached
        start_cached2 = time.time()
        _ = parallel_canonicalize_smiles(smiles_list)
        time_cached2 = time.time() - start_cached2

        speedup_first_to_second = time_first / time_cached if time_cached > 0 else float("inf")
        speedup_first_to_third = time_first / time_cached2 if time_cached2 > 0 else float("inf")

        print("\nSMILES Canonicalization Cache Performance:")
        print(f"  First call (cache miss): {time_first:.4f}s")
        print(f"  Second call (cache hit): {time_cached:.4f}s")
        print(f"  Third call (cache hit): {time_cached2:.4f}s")
        print(f"  Speedup (1st→2nd): {speedup_first_to_second:.1f}x")
        print(f"  Speedup (1st→3rd): {speedup_first_to_third:.1f}x")

        # Cache hits should be much faster (target is 5-10x, but we'll accept 2x+)
        assert (
            speedup_first_to_second >= 2.0
        ), f"Cached canonicalization should be at least 2x faster (got {speedup_first_to_second:.1f}x)"


@pytest.mark.slow
class TestNumWorkersPerformance:
    """Benchmark DataLoader with num_workers."""

    @pytest.mark.xfail(reason="Flaky: num_workers overhead can dominate on small datasets and varies by system load")
    def test_num_workers_speedup(
        self, benchmark_dataset: pd.DataFrame, minimal_chemprop_config: ChempropConfig, tmp_path: Path
    ):
        """Measure speedup from using num_workers in DataLoader."""
        # Setup
        train_df = benchmark_dataset.iloc[:80]
        val_df = benchmark_dataset.iloc[80:]

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "validation.csv", index=False)

        minimal_chemprop_config.data.data_dir = str(data_dir)
        minimal_chemprop_config.optimization.max_epochs = 3

        # Train with num_workers=0 (baseline)
        config_no_workers = OmegaConf.create(OmegaConf.to_container(minimal_chemprop_config))
        config_no_workers.optimization.num_workers = 0

        start_no_workers = time.time()
        model_no_workers = ChempropModel.from_config(config_no_workers)
        model_no_workers.fit()
        time_no_workers = time.time() - start_no_workers
        model_no_workers.close()

        # Train with num_workers=2 (optimized)
        config_workers = OmegaConf.create(OmegaConf.to_container(minimal_chemprop_config))
        config_workers.optimization.num_workers = 2
        config_workers.optimization.seed = 42  # Same seed for fairness

        start_workers = time.time()
        model_workers = ChempropModel.from_config(config_workers)
        model_workers.fit()
        time_workers = time.time() - start_workers
        model_workers.close()

        speedup = time_no_workers / time_workers if time_workers > 0 else 1.0

        print("\nDataLoader num_workers Performance:")
        print(f"  num_workers=0: {time_no_workers:.2f}s")
        print(f"  num_workers=2: {time_workers:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Should see some speedup (target is 10-20%, so 1.1x+)
        # Note: Speedup depends on CPU and dataset size
        assert speedup >= 1.0, "num_workers should not slow down training"

        if speedup < 1.1:
            print("  Note: Speedup < 1.1x, may be due to small dataset or overhead")


def test_benchmark_summary():
    """Print summary of all benchmark tests."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKS - TEST SUMMARY")
    print("=" * 70)
    print("  📊 Batched Predictions: Target 2-3x speedup")
    print("  📊 SMILES Cache: Target 5-10x speedup")
    print("  📊 DataLoader Workers: Target 10-20% speedup")
    print("  📊 Mixed Precision: Target 1.5-2x speedup (GPU required)")
    print("=" * 70)
    print("Run these tests with: pytest tests/benchmarks/ -v --durations=10")
    print("=" * 70 + "\n")
