"""Performance Profiling Utilities
=================================

Comprehensive profiling system for tracking runtime performance across
training, HPO, and ensemble workflows. Provides visibility into phase-level
timing breakdowns with MLflow integration.

Contents
--------
Classes
^^^^^^^
* :class:`PhaseTimer` - Context manager for timing individual phases
* :class:`TrainingProfiler` - Full training profiler with nested phase support
* :class:`ProfilingCallback` - PyTorch Lightning callback for epoch-level profiling

Functions
^^^^^^^^^
* :func:`profile_phase` - Decorator for profiling function execution
* :func:`format_duration` - Human-readable duration formatting
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

import numpy as np

if TYPE_CHECKING:
    import mlflow
    from lightning import pytorch as pl
    from mlflow import MlflowClient

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class TrainingPhase(str, Enum):
    """Standard phases in model training workflow."""

    # Pre-training phases
    INIT = "init"
    CONFIG_LOAD = "config_load"
    DATA_LOAD = "data_load"
    DATA_PREPROCESSING = "data_preprocessing"
    SMILES_CANONICALIZATION = "smiles_canonicalization"
    FEATURE_GENERATION = "feature_generation"
    DATASET_CREATION = "dataset_creation"
    TARGET_SCALING = "target_scaling"
    DATALOADER_CREATION = "dataloader_creation"
    MODEL_INIT = "model_init"
    CHECKPOINT_LOAD = "checkpoint_load"
    TRAINER_SETUP = "trainer_setup"
    MLFLOW_INIT = "mlflow_init"

    # Training phases
    TRAINING_TOTAL = "training_total"
    EPOCH = "epoch"
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    OPTIMIZER_STEP = "optimizer_step"
    VALIDATION = "validation"
    EARLY_STOPPING_CHECK = "early_stopping_check"

    # Post-training phases
    BEST_CHECKPOINT_LOAD = "best_checkpoint_load"
    EVALUATION = "evaluation"
    METRICS_COMPUTATION = "metrics_computation"
    PLOT_GENERATION = "plot_generation"
    ARTIFACT_LOGGING = "artifact_logging"
    MODEL_SAVE = "model_save"
    CLEANUP = "cleanup"

    # HPO-specific phases
    HPO_TOTAL = "hpo_total"
    HPO_SEARCH_SPACE_BUILD = "hpo_search_space_build"
    HPO_RAY_INIT = "hpo_ray_init"
    HPO_TRIAL = "hpo_trial"
    HPO_RESULTS_AGGREGATION = "hpo_results_aggregation"

    # Ensemble-specific phases
    ENSEMBLE_TOTAL = "ensemble_total"
    ENSEMBLE_SPLIT_DISCOVERY = "ensemble_split_discovery"
    DATA_DISCOVERY = "data_discovery"
    ENSEMBLE_MODEL_TRAIN = "ensemble_model_train"
    ENSEMBLE_PREDICTION = "ensemble_prediction"
    ENSEMBLE_AGGREGATION = "ensemble_aggregation"
    ENSEMBLE_OUTPUT = "ensemble_output"


@dataclass
class TimingRecord:
    """Record of a single timing measurement."""

    phase: str
    duration_seconds: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseStats:
    """Aggregated statistics for a phase across multiple measurements."""

    phase: str
    count: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0
    durations: List[float] = field(default_factory=list)

    @property
    def mean_seconds(self) -> float:
        """Average duration in seconds."""
        return self.total_seconds / self.count if self.count > 0 else 0.0

    @property
    def std_seconds(self) -> float:
        """Standard deviation of durations."""
        if self.count < 2:
            return 0.0
        return float(np.std(self.durations))

    def add_timing(self, duration: float) -> None:
        """Add a timing measurement."""
        self.count += 1
        self.total_seconds += duration
        self.min_seconds = min(self.min_seconds, duration)
        self.max_seconds = max(self.max_seconds, duration)
        self.durations.append(duration)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string (e.g., "1h 23m 45.6s", "45.6s", "123.4ms").
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


class PhaseTimer:
    """Context manager for timing a single phase.

    Parameters
    ----------
    phase : str or TrainingPhase
        Name of the phase being timed.
    profiler : TrainingProfiler, optional
        Parent profiler to record timing to.
    metadata : dict, optional
        Additional metadata to attach to the timing record.
    log_on_exit : bool, default=True
        Whether to log timing on context exit.

    Examples
    --------
    >>> with PhaseTimer("data_load") as timer:
    ...     data = load_data()
    >>> print(f"Data loading took {timer.duration:.2f}s")
    """

    def __init__(
        self,
        phase: str | TrainingPhase,
        profiler: Optional["TrainingProfiler"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_on_exit: bool = True,
    ) -> None:
        self.phase = phase.value if isinstance(phase, TrainingPhase) else phase
        self.profiler = profiler
        self.metadata = metadata or {}
        self.log_on_exit = log_on_exit
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and record."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.profiler is not None:
            self.profiler.record_timing(self.phase, self.duration, self.metadata)

        if self.log_on_exit:
            logger.debug(
                "Phase '%s' completed in %s",
                self.phase(self.duration),
            )


class TrainingProfiler:
    """Comprehensive profiler for tracking training performance.

    Supports nested phases, MLflow integration, and summary generation.

    Parameters
    ----------
    name : str, default="training"
        Name identifier for this profiler instance.
    mlflow_client : MlflowClient, optional
        MLflow client for logging metrics.
    mlflow_run_id : str, optional
        MLflow run ID to log metrics to.
    enabled : bool, default=True
        Whether profiling is enabled.

    Examples
    --------
    >>> profiler = TrainingProfiler("chemprop_training")
    >>> with profiler.phase("data_load"):
    ...     df = pd.read_csv("train.csv")
    >>> with profiler.phase("model_init"):
    ...     model = ChempropModel(config)
    >>> profiler.print_summary()
    """

    def __init__(
        self,
        name: str = "training",
        mlflow_client: Optional["MlflowClient"] = None,
        mlflow_run_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.mlflow_client = mlflow_client
        self.mlflow_run_id = mlflow_run_id
        self.enabled = enabled

        self._timings: List[TimingRecord] = []
        self._phase_stats: Dict[str, PhaseStats] = defaultdict(lambda: PhaseStats(phase=""))
        self._active_phases: List[str] = []
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def start(self) -> "TrainingProfiler":
        """Mark the start of the profiled workflow."""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> "TrainingProfiler":
        """Mark the end of the profiled workflow."""
        self._end_time = time.perf_counter()
        return self

    @property
    def total_duration(self) -> float:
        """Total duration from start to stop in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @contextmanager
    def phase(
        self,
        phase: str | TrainingPhase,
        metadata: Optional[Dict[str, Any]] = None,
        log_on_exit: bool = True,
    ):
        """Context manager for timing a phase.

        Parameters
        ----------
        phase : str or TrainingPhase
            Phase identifier.
        metadata : dict, optional
            Additional metadata.
        log_on_exit : bool, default=True
            Whether to log on exit.

        Yields
        ------
        PhaseTimer
            Timer instance for the phase.
        """
        if not self.enabled:
            yield None
            return

        timer = PhaseTimer(phase, self, metadata, log_on_exit)
        phase_name = phase.value if isinstance(phase, TrainingPhase) else phase

        with self._lock:
            self._active_phases.append(phase_name)

        try:
            with timer:
                yield timer
        finally:
            with self._lock:
                if phase_name in self._active_phases:
                    self._active_phases.remove(phase_name)

    def record_timing(
        self,
        phase: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a timing measurement.

        Parameters
        ----------
        phase : str
            Phase identifier.
        duration : float
            Duration in seconds.
        metadata : dict, optional
            Additional metadata.
        """
        if not self.enabled:
            return

        record = TimingRecord(
            phase=phase,
            duration_seconds=duration,
            start_time=time.perf_counter() - duration,
            end_time=time.perf_counter(),
            metadata=metadata or {},
        )

        with self._lock:
            self._timings.append(record)

            if phase not in self._phase_stats:
                self._phase_stats[phase] = PhaseStats(phase=phase)
            self._phase_stats[phase].add_timing(duration)

    def get_phase_stats(self, phase: str) -> Optional[PhaseStats]:
        """Get statistics for a specific phase.

        Parameters
        ----------
        phase : str
            Phase identifier.

        Returns
        -------
        PhaseStats or None
            Statistics for the phase, or None if not recorded.
        """
        return self._phase_stats.get(phase)

    def get_all_stats(self) -> Dict[str, PhaseStats]:
        """Get statistics for all phases.

        Returns
        -------
        dict
            Mapping of phase names to statistics.
        """
        return dict(self._phase_stats)

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get a summary dictionary suitable for logging.

        Returns
        -------
        dict
            Summary with phase timings and statistics.
        """
        summary: Dict[str, Any] = {
            "profiler_name": self.name,
            "total_duration_seconds": self.total_duration,
            "total_duration_formatted": format_duration(self.total_duration),
            "phases": {},
        }

        for phase, stats in sorted(self._phase_stats.items(), key=lambda x: -x[1].total_seconds):
            pct = (stats.total_seconds / self.total_duration * 100) if self.total_duration > 0 else 0
            summary["phases"][phase] = {
                "count": stats.count,
                "total_seconds": stats.total_seconds,
                "mean_seconds": stats.mean_seconds,
                "min_seconds": stats.min_seconds if stats.count > 0 else 0,
                "max_seconds": stats.max_seconds,
                "std_seconds": stats.std_seconds,
                "percentage_of_total": pct,
            }

        return summary

    def print_summary(self, top_n: int = 20) -> None:
        """Print a formatted summary of profiling results.

        Parameters
        ----------
        top_n : int, default=20
            Number of top phases to display.
        """
        if not self._phase_stats:
            logger.info("No profiling data recorded")
            return

        print(f"\n{'=' * 80}")
        print(f" PROFILING SUMMARY: {self.name}")
        print(f"{'=' * 80}")
        print(f" Total Duration: {format_duration(self.total_duration)}")
        print(f"{'=' * 80}")
        print(f"{'Phase':<40} {'Count':>6} {'Total':>12} {'Mean':>12} {'%':>8}")
        print(f"{'-' * 80}")

        sorted_phases = sorted(self._phase_stats.items(), key=lambda x: -x[1].total_seconds)
        for phase, stats in sorted_phases[:top_n]:
            pct = (stats.total_seconds / self.total_duration * 100) if self.total_duration > 0 else 0
            print(
                f"{phase:<40} {stats.count:>6} {format_duration(stats.total_seconds):>12} "
                f"{format_duration(stats.mean_seconds):>12} {pct:>7.1f}%"
            )

        if len(sorted_phases) > top_n:
            print(f"... and {len(sorted_phases) - top_n} more phases")

        print(f"{'=' * 80}\n")

    def log_to_mlflow(
        self,
        prefix: str = "profiling",
        client: Optional["MlflowClient"] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Log profiling metrics to MLflow.

        Parameters
        ----------
        prefix : str, default="profiling"
            Prefix for metric names.
        client : MlflowClient, optional
            MLflow client. Uses instance client if not provided.
        run_id : str, optional
            MLflow run ID. Uses instance run_id if not provided.
        """
        client = client or self.mlflow_client
        run_id = run_id or self.mlflow_run_id

        if client is None or run_id is None:
            logger.debug("MLflow client or run_id not available, skipping profiling log")
            return

        try:
            # Log total duration
            client.log_metric(run_id, f"{prefix}.total_seconds", self.total_duration)

            # Log per-phase metrics
            for phase, stats in self._phase_stats.items():
                safe_phase = phase.replace(".", "_").replace("-", "_")
                client.log_metric(run_id, f"{prefix}.{safe_phase}.total_seconds", stats.total_seconds)
                client.log_metric(run_id, f"{prefix}.{safe_phase}.mean_seconds", stats.mean_seconds)
                client.log_metric(run_id, f"{prefix}.{safe_phase}.count", stats.count)

                if self.total_duration > 0:
                    pct = stats.total_seconds / self.total_duration * 100
                    client.log_metric(run_id, f"{prefix}.{safe_phase}.percentage", pct)

            logger.debug("Logged profiling metrics to MLflow run %s", run_id)

        except Exception as e:
            logger.warning("Failed to log profiling metrics to MLflow: %s", e)

    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._timings.clear()
            self._phase_stats.clear()
            self._active_phases.clear()
            self._start_time = None
            self._end_time = None


def profile_phase(
    phase: str | TrainingPhase,
    profiler_attr: str = "_profiler",
) -> Callable[[F], F]:
    """Decorator to profile a function as a phase.

    Parameters
    ----------
    phase : str or TrainingPhase
        Phase identifier.
    profiler_attr : str, default="_profiler"
        Name of the profiler attribute on self.

    Returns
    -------
    Callable
        Decorated function.

    Examples
    --------
    >>> class MyModel:
    ...     def __init__(self):
    ...         self._profiler = TrainingProfiler()
    ...
    ...     @profile_phase("data_load")
    ...     def load_data(self):
    ...         return pd.read_csv("data.csv")
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            profiler = getattr(self, profiler_attr, None)
            if profiler is not None and isinstance(profiler, TrainingProfiler) and profiler.enabled:
                with profiler.phase(phase):
                    return func(self, *args, **kwargs)
            return func(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


class ProfilingCallback:
    """PyTorch Lightning callback for epoch-level profiling.

    Tracks timing for training epochs, validation, and other training events.

    Parameters
    ----------
    profiler : TrainingProfiler
        Parent profiler to record timings to.
    log_every_n_epochs : int, default=1
        How often to log epoch timings.

    Examples
    --------
    >>> profiler = TrainingProfiler()
    >>> callback = ProfilingCallback(profiler)
    >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        profiler: TrainingProfiler,
        log_every_n_epochs: int = 1,
    ) -> None:
        self.profiler = profiler
        self.log_every_n_epochs = log_every_n_epochs
        self._epoch_start_time: Optional[float] = None
        self._train_epoch_start: Optional[float] = None
        self._val_epoch_start: Optional[float] = None

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Record start of training epoch."""
        self._train_epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Record end of training epoch."""
        if self._train_epoch_start is not None:
            duration = time.perf_counter() - self._train_epoch_start
            epoch = trainer.current_epoch
            if epoch % self.log_every_n_epochs == 0:
                self.profiler.record_timing(
                    f"train_epoch_{epoch}",
                    duration,
                    {"epoch": epoch},
                )
            self.profiler.record_timing(TrainingPhase.EPOCH.value, duration, {"epoch": epoch, "type": "train"})
            self._train_epoch_start = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Record start of validation epoch."""
        self._val_epoch_start = time.perf_counter()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Record end of validation epoch."""
        if self._val_epoch_start is not None:
            duration = time.perf_counter() - self._val_epoch_start
            epoch = trainer.current_epoch
            self.profiler.record_timing(TrainingPhase.VALIDATION.value, duration, {"epoch": epoch})
            self._val_epoch_start = None


class LightningProfilingCallback:
    """PyTorch Lightning Callback class for profiling.

    This is a proper Lightning callback that inherits from pl.Callback.
    Import this in modules that use Lightning.
    """

    pass  # Placeholder - actual implementation requires lightning import


def create_lightning_profiling_callback(profiler: TrainingProfiler, log_every_n_epochs: int = 1) -> Any:
    """Create a PyTorch Lightning callback for profiling.

    Parameters
    ----------
    profiler : TrainingProfiler
        Parent profiler.
    log_every_n_epochs : int, default=1
        How often to log epoch timings.

    Returns
    -------
    pl.Callback
        Lightning callback instance.
    """
    from lightning import pytorch as pl

    class _ProfilingCallback(pl.Callback):
        """Lightning callback for profiling training phases."""

        def __init__(self) -> None:
            super().__init__()
            self._profiler = profiler
            self._log_every = log_every_n_epochs
            self._train_epoch_start: Optional[float] = None
            self._val_epoch_start: Optional[float] = None
            self._train_batch_start: Optional[float] = None

        def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Record start of training."""
            self._profiler.start()

        def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Record end of training."""
            self._profiler.stop()

        def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Record start of training epoch."""
            self._train_epoch_start = time.perf_counter()

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Record end of training epoch."""
            if self._train_epoch_start is not None:
                duration = time.perf_counter() - self._train_epoch_start
                epoch = trainer.current_epoch
                if epoch % self._log_every == 0:
                    self._profiler.record_timing(
                        f"train_epoch_{epoch}",
                        duration,
                        {"epoch": epoch},
                    )
                self._profiler.record_timing(
                    TrainingPhase.EPOCH.value,
                    duration,
                    {"epoch": epoch, "type": "train"},
                )
                self._train_epoch_start = None

        def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Record start of validation."""
            self._val_epoch_start = time.perf_counter()

        def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            """Record end of validation."""
            if self._val_epoch_start is not None:
                duration = time.perf_counter() - self._val_epoch_start
                self._profiler.record_timing(
                    TrainingPhase.VALIDATION.value,
                    duration,
                    {"epoch": trainer.current_epoch},
                )
                self._val_epoch_start = None

        def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Any,
            batch_idx: int,
        ) -> None:
            """Record start of training batch."""
            self._train_batch_start = time.perf_counter()

        def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            """Record end of training batch (samples for detailed profiling)."""
            # Only record batch timing occasionally to avoid overhead
            if self._train_batch_start is not None and batch_idx % 100 == 0:
                duration = time.perf_counter() - self._train_batch_start
                self._profiler.record_timing(
                    "train_batch",
                    duration,
                    {"batch_idx": batch_idx, "epoch": trainer.current_epoch},
                )
            self._train_batch_start = None

    return _ProfilingCallback()


# Global profiler for convenience (can be used across modules)
_global_profiler: Optional[TrainingProfiler] = None


def get_global_profiler() -> TrainingProfiler:
    """Get or create the global profiler instance.

    Returns
    -------
    TrainingProfiler
        Global profiler instance.
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = TrainingProfiler(name="global")
    return _global_profiler


def set_global_profiler(profiler: TrainingProfiler) -> None:
    """Set the global profiler instance.

    Parameters
    ----------
    profiler : TrainingProfiler
        Profiler to set as global.
    """
    global _global_profiler
    _global_profiler = profiler


@contextmanager
def timed_phase(phase: str | TrainingPhase, metadata: Optional[Dict[str, Any]] = None):
    """Convenience context manager using global profiler.

    Parameters
    ----------
    phase : str or TrainingPhase
        Phase identifier.
    metadata : dict, optional
        Additional metadata.

    Yields
    ------
    PhaseTimer
        Timer instance.

    Examples
    --------
    >>> with timed_phase("data_load"):
    ...     data = load_data()
    """
    profiler = get_global_profiler()
    with profiler.phase(phase, metadata) as timer:
        yield timer
