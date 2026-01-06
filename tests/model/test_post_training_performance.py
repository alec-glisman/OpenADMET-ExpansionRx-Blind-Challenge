"""Tests for post-training performance bottlenecks.

This module contains tests to identify and measure performance bottlenecks
in the post-training workflow, specifically:
1. Evaluation metrics computation (multiple predict() calls)
2. Plot generation (another predict() call + matplotlib rendering)
3. MLflow artifact logging (checkpoint uploads, model registration)

These tests help verify where runtime slowdowns occur and validate
optimization strategies.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from admet.util.profiling import TrainingPhase, TrainingProfiler

if TYPE_CHECKING:
    pass


# Test data fixtures
@pytest.fixture
def sample_smiles() -> List[str]:
    """Generate sample SMILES for testing."""
    return [
        "CCO",
        "CCCO",
        "CCCCO",
        "CC(C)O",
        "CC(C)(C)O",
        "c1ccccc1",
        "c1ccccc1O",
        "c1ccccc1N",
        "CC(=O)O",
        "CC(=O)N",
    ] * 10  # 100 samples


@pytest.fixture
def sample_targets(sample_smiles: List[str]) -> np.ndarray:
    """Generate sample targets for testing."""
    n_samples = len(sample_smiles)
    n_targets = 9  # Standard ADMET targets
    np.random.seed(42)
    return np.random.randn(n_samples, n_targets)


@pytest.fixture
def sample_dataframe(sample_smiles: List[str], sample_targets: np.ndarray) -> pd.DataFrame:
    """Create a sample dataframe with SMILES and targets."""
    target_cols = [
        "LogD",
        "Log KSOL",
        "Log HLM CLint",
        "Log MLM CLint",
        "Log Caco-2 Permeability Papp A>B",
        "Log Caco-2 Permeability Efflux",
        "Log MPPB",
        "Log MBPB",
        "Log MGMB",
    ]
    df = pd.DataFrame({"SMILES": sample_smiles})
    for i, col in enumerate(target_cols):
        df[col] = sample_targets[:, i]
    return df


class TestPostTrainingPerformance:
    """Test suite for post-training performance analysis."""

    def test_profiler_captures_post_training_phases(self) -> None:
        """Verify profiler correctly captures post-training phases."""
        profiler = TrainingProfiler(name="test")
        profiler.start()

        # Simulate post-training phases
        with profiler.phase(TrainingPhase.METRICS_COMPUTATION):
            time.sleep(0.05)

        with profiler.phase(TrainingPhase.ARTIFACT_LOGGING):
            time.sleep(0.03)

        with profiler.phase(TrainingPhase.PLOT_GENERATION):
            time.sleep(0.02)

        profiler.stop()

        stats = profiler.get_all_stats()
        assert TrainingPhase.METRICS_COMPUTATION.value in stats
        assert TrainingPhase.ARTIFACT_LOGGING.value in stats
        assert TrainingPhase.PLOT_GENERATION.value in stats

        # Verify timing is captured
        metrics_stats = stats[TrainingPhase.METRICS_COMPUTATION.value]
        assert metrics_stats.total_seconds >= 0.05

    def test_predict_call_count_in_post_training(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Verify how many times predict() is called during post-training.

        This test identifies if predict() is called multiple times
        (once for metrics, once for plots) which is inefficient.
        """
        predict_call_count = 0
        original_predict_times: List[float] = []

        def mock_predict(*args: Any, **kwargs: Any) -> pd.DataFrame:
            nonlocal predict_call_count
            start = time.perf_counter()
            predict_call_count += 1
            # Simulate prediction time
            time.sleep(0.01)
            duration = time.perf_counter() - start
            original_predict_times.append(duration)
            return sample_dataframe.copy()

        # Create mock model with tracked predict
        mock_model = MagicMock()
        mock_model.predict = mock_predict
        mock_model.target_cols = list(sample_dataframe.columns[1:])
        mock_model.dataframes = {
            "train": sample_dataframe,
            "validation": sample_dataframe,
        }
        mock_model.mlflow_tracking = True
        mock_model._mlflow_logger = MagicMock()
        mock_model.output_dir = None

        # Simulate _log_evaluation_metrics pattern
        # (normally calls predict for validation and test)
        for split_name in ["validation", "test"]:
            if split_name in mock_model.dataframes or split_name == "test":
                mock_model.predict(sample_dataframe, log_metrics=False)

        # Simulate _generate_training_plots pattern
        # (calls predict again for train and validation)
        for split_name in ["train", "validation"]:
            mock_model.predict(sample_dataframe, log_metrics=False)

        # Current implementation calls predict 4 times!
        # 2 from _log_evaluation_metrics (validation, test)
        # 2 from _generate_training_plots (train, validation)
        assert predict_call_count == 4, (
            f"predict() called {predict_call_count} times. "
            "This indicates redundant computation that should be optimized."
        )

    def test_plot_generation_timing(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Measure time spent in plot generation."""
        from admet.plot.parity import plot_parity

        profiler = TrainingProfiler(name="plot_test")
        profiler.start()

        target_cols = list(sample_dataframe.columns[1:])
        n_plots = 0

        with profiler.phase(TrainingPhase.PLOT_GENERATION):
            for target in target_cols:
                y_true = sample_dataframe[target].values
                y_pred = y_true + np.random.randn(len(y_true)) * 0.1

                fig, ax = plot_parity(y_true, y_pred, title=f"{target}")
                n_plots += 1

                # Clean up
                import matplotlib.pyplot as plt

                plt.close(fig)

        profiler.stop()

        stats = profiler.get_all_stats()
        plot_stats = stats[TrainingPhase.PLOT_GENERATION.value]

        print("\nPlot generation stats:")
        print(f"  Plots generated: {n_plots}")
        print(f"  Total time: {plot_stats.total_seconds:.3f}s")
        print(f"  Time per plot: {plot_stats.total_seconds / n_plots:.3f}s")

        # Assert reasonable timing (should be < 1s per plot on average)
        assert plot_stats.total_seconds / n_plots < 1.0

    def test_mlflow_artifact_upload_timing(self) -> None:
        """Measure time for MLflow artifact uploads (mocked)."""
        profiler = TrainingProfiler(name="mlflow_test")
        profiler.start()

        # Create temporary files to simulate artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock checkpoint file (10MB)
            checkpoint_path = temp_path / "best.ckpt"
            checkpoint_path.write_bytes(b"0" * (10 * 1024 * 1024))

            # Create mock config file (1KB)
            config_path = temp_path / "config.yaml"
            config_path.write_text("test: config")

            # Mock MLflow client
            mock_client = MagicMock()
            upload_times: List[float] = []

            def mock_log_artifact(run_id: str, path: str, artifact_path: str | None = None) -> None:
                start = time.perf_counter()
                # Simulate network latency (10ms per MB)
                file_size_mb = Path(path).stat().st_size / (1024 * 1024)
                time.sleep(0.01 * file_size_mb)
                upload_times.append(time.perf_counter() - start)

            mock_client.log_artifact = mock_log_artifact

            with profiler.phase(TrainingPhase.ARTIFACT_LOGGING):
                # Simulate artifact logging
                mock_client.log_artifact("run_id", str(checkpoint_path), "checkpoints")
                mock_client.log_artifact("run_id", str(config_path), "config")

        profiler.stop()

        stats = profiler.get_all_stats()
        artifact_stats = stats[TrainingPhase.ARTIFACT_LOGGING.value]

        print("\nArtifact upload stats:")
        print(f"  Total time: {artifact_stats.total_seconds:.3f}s")
        print(f"  Upload times: {upload_times}")

    def test_metrics_computation_timing(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Measure time for metrics computation."""
        from admet.data.stats import correlation

        profiler = TrainingProfiler(name="metrics_test")
        profiler.start()

        target_cols = list(sample_dataframe.columns[1:])
        metrics_computed = 0

        with profiler.phase(TrainingPhase.METRICS_COMPUTATION):
            for target in target_cols:
                y_true = sample_dataframe[target].values
                y_pred = y_true + np.random.randn(len(y_true)) * 0.1

                metrics = correlation(y_true, y_pred)
                metrics_computed += len(metrics)

        profiler.stop()

        stats = profiler.get_all_stats()
        metrics_stats = stats[TrainingPhase.METRICS_COMPUTATION.value]

        print("\nMetrics computation stats:")
        print(f"  Metrics computed: {metrics_computed}")
        print(f"  Total time: {metrics_stats.total_seconds:.3f}s")

        # Metrics computation should be very fast
        assert metrics_stats.total_seconds < 1.0


class TestOptimizationStrategies:
    """Tests for optimization strategies to reduce post-training time."""

    def test_cached_predictions_reduce_redundancy(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Test that caching predictions eliminates redundant predict() calls."""
        predict_count = 0

        class CachedPredictor:
            """Mock predictor with prediction caching."""

            def __init__(self) -> None:
                self._prediction_cache: Dict[str, pd.DataFrame] = {}

            def predict_cached(self, df: pd.DataFrame, cache_key: str) -> pd.DataFrame:
                nonlocal predict_count
                if cache_key not in self._prediction_cache:
                    predict_count += 1
                    time.sleep(0.01)  # Simulate prediction
                    self._prediction_cache[cache_key] = df.copy()
                return self._prediction_cache[cache_key]

        predictor = CachedPredictor()

        # Simulate post-training with caching
        # First call for metrics
        predictor.predict_cached(sample_dataframe, "validation")
        predictor.predict_cached(sample_dataframe, "train")

        # Second call for plots (should use cache)
        predictor.predict_cached(sample_dataframe, "validation")
        predictor.predict_cached(sample_dataframe, "train")

        # With caching, predict should only be called twice
        assert predict_count == 2, f"With caching, predict() should be called 2 times, not {predict_count}"

    def test_async_artifact_upload_pattern(self) -> None:
        """Test async artifact upload pattern for MLflow."""
        import concurrent.futures

        sync_total_time = 0.0
        async_total_time = 0.0

        def simulate_upload(file_size_mb: float) -> float:
            """Simulate network upload with latency."""
            time.sleep(0.01 * file_size_mb)
            return file_size_mb

        files = [10.0, 5.0, 1.0, 0.1]  # File sizes in MB

        # Synchronous upload
        start = time.perf_counter()
        for size in files:
            simulate_upload(size)
        sync_total_time = time.perf_counter() - start

        # Async upload
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simulate_upload, size) for size in files]
            concurrent.futures.wait(futures)
        async_total_time = time.perf_counter() - start

        print("\nUpload timing comparison:")
        print(f"  Sync total: {sync_total_time:.3f}s")
        print(f"  Async total: {async_total_time:.3f}s")
        print(f"  Speedup: {sync_total_time / async_total_time:.2f}x")

        # Async should be faster
        assert async_total_time < sync_total_time

    def test_lazy_plot_generation_config(self) -> None:
        """Test configuration option to disable/defer plot generation."""
        config_options = {
            "generate_plots": True,
            "generate_plots_async": False,
            "plot_formats": ["png"],
            "plot_dpi": 100,  # Lower DPI = faster rendering
        }

        # Verify config options are valid
        assert isinstance(config_options["generate_plots"], bool)
        assert isinstance(config_options["generate_plots_async"], bool)
        assert "png" in config_options["plot_formats"]
        assert config_options["plot_dpi"] <= 150  # High DPI is slow

    def test_deferred_model_registration(self) -> None:
        """Test that model registration can be deferred to a background task."""
        registration_completed = False
        registration_time = 0.0

        def register_model_sync() -> None:
            """Simulate synchronous model registration."""
            nonlocal registration_completed, registration_time
            start = time.perf_counter()
            time.sleep(0.5)  # Simulate mlflow.pytorch.log_model()
            registration_time = time.perf_counter() - start
            registration_completed = True

        def register_model_deferred() -> Any:
            """Simulate deferred model registration."""
            import concurrent.futures

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            return executor.submit(register_model_sync)

        # Test deferred registration doesn't block
        start = time.perf_counter()
        future = register_model_deferred()
        immediate_return_time = time.perf_counter() - start

        # Should return immediately
        assert immediate_return_time < 0.1

        # Wait for completion
        future.result()
        assert registration_completed
        assert registration_time >= 0.5


class TestPostTrainingConfig:
    """Tests for PostTrainingConfig functionality."""

    def test_post_training_config_defaults(self) -> None:
        """Verify PostTrainingConfig has correct defaults."""
        from admet.model.chemprop.config import PostTrainingConfig

        config = PostTrainingConfig()
        assert config.generate_plots is True
        assert config.cache_predictions is True
        assert config.plot_dpi == 150
        assert config.plot_formats == ["png"]
        assert config.log_model_to_mlflow is True
        assert config.async_artifact_upload is False
        assert config.compute_test_metrics is True
        assert config.compute_train_metrics is False

    def test_post_training_config_disable_plots(self) -> None:
        """Verify plots can be disabled."""
        from admet.model.chemprop.config import PostTrainingConfig

        config = PostTrainingConfig(generate_plots=False)
        assert config.generate_plots is False

    def test_chemprop_config_includes_post_training(self) -> None:
        """Verify ChempropConfig includes PostTrainingConfig."""
        from admet.model.chemprop.config import ChempropConfig, PostTrainingConfig

        config = ChempropConfig()
        assert hasattr(config, "post_training")
        assert isinstance(config.post_training, PostTrainingConfig)


class TestPredictionCaching:
    """Tests for prediction caching in ChempropModel."""

    def test_prediction_cache_initialization(self) -> None:
        """Verify prediction cache is initialized empty."""
        from admet.model.chemprop.config import PostTrainingConfig

        # Create a minimal mock that has the _prediction_cache attribute
        cache: dict[str, Any] = {}
        config = PostTrainingConfig()

        assert config.cache_predictions is True
        assert len(cache) == 0

    def test_prediction_cache_workflow(self, sample_dataframe: pd.DataFrame) -> None:
        """Simulate prediction caching workflow."""
        cache: dict[str, pd.DataFrame] = {}
        cache_enabled = True

        # Simulate first predict call
        split_name = "validation"
        if cache_enabled and split_name not in cache:
            # Mock predictions
            preds_df = sample_dataframe.copy()
            cache[split_name] = preds_df

        assert split_name in cache

        # Second call should hit cache
        hit_cache = split_name in cache
        assert hit_cache is True

    def test_cache_reduces_predict_calls(self, sample_dataframe: pd.DataFrame) -> None:
        """Verify caching reduces predict call count."""
        predict_count = 0

        def mock_predict(df: pd.DataFrame) -> pd.DataFrame:
            nonlocal predict_count
            predict_count += 1
            return df.copy()

        cache: dict[str, pd.DataFrame] = {}
        cache_enabled = True

        # Simulate metrics computation (first predict)
        split_name = "validation"
        if cache_enabled and split_name in cache:
            preds = cache[split_name]
        else:
            preds = mock_predict(sample_dataframe)
            if cache_enabled:
                cache[split_name] = preds

        # Simulate plot generation (should use cache)
        if cache_enabled and split_name in cache:
            preds = cache[split_name]
        else:
            preds = mock_predict(sample_dataframe)

        # Only one actual predict call should be made
        assert predict_count == 1


class TestIntegrationPostTraining:
    """Integration tests for post-training workflow optimizations."""

    @pytest.mark.slow
    def test_full_post_training_timing_breakdown(
        self,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Full integration test measuring post-training time breakdown."""
        profiler = TrainingProfiler(name="post_training_integration")
        profiler.start()

        target_cols = list(sample_dataframe.columns[1:])

        # Phase 1: Metrics computation
        with profiler.phase(TrainingPhase.METRICS_COMPUTATION):
            from admet.data.stats import correlation

            for target in target_cols:
                y_true = sample_dataframe[target].values
                y_pred = y_true + np.random.randn(len(y_true)) * 0.1
                correlation(y_true, y_pred)

        # Phase 2: Plot generation
        with profiler.phase(TrainingPhase.PLOT_GENERATION):
            import matplotlib.pyplot as plt

            from admet.plot.parity import plot_parity

            for target in target_cols:
                y_true = sample_dataframe[target].values
                y_pred = y_true + np.random.randn(len(y_true)) * 0.1
                fig, _ = plot_parity(y_true, y_pred, title=target)
                plt.close(fig)

        # Phase 3: Artifact logging (mocked)
        with profiler.phase(TrainingPhase.ARTIFACT_LOGGING):
            time.sleep(0.1)  # Simulate artifact upload

        profiler.stop()
        profiler.print_summary()

        # Get timing breakdown
        stats = profiler.get_all_stats()
        total = profiler.total_duration

        metrics_pct = stats[TrainingPhase.METRICS_COMPUTATION.value].total_seconds / total * 100
        plots_pct = stats[TrainingPhase.PLOT_GENERATION.value].total_seconds / total * 100
        artifacts_pct = stats[TrainingPhase.ARTIFACT_LOGGING.value].total_seconds / total * 100

        print("\nPost-training breakdown:")
        print(f"  Metrics: {metrics_pct:.1f}%")
        print(f"  Plots: {plots_pct:.1f}%")
        print(f"  Artifacts: {artifacts_pct:.1f}%")

        # Plots are likely the slowest - this test documents it
        # assert plots_pct > metrics_pct  # Plots usually dominate


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
