"""Ray Tune logging utilities for capturing and managing output.

This module provides utilities to configure Ray's logging infrastructure,
redirect verbose output to files, upload logs as MLflow artifacts, and
track ensemble training progress with minimal terminal output.

Key Components
--------------
RayLogManager
    Context manager that configures Ray environment variables for logging,
    collects trial logs after experiment completion, compresses them, and
    uploads as MLflow artifacts. Handles cleanup and interrupt signals.

QuietProgressReporter
    Minimal progress reporter that displays only completion summaries
    instead of per-trial details, keeping terminal output clean.

LogArtifactCallback
    Ray Tune callback that collects logs from all trials and uploads
    them to MLflow as a compressed artifact after experiment completion.

EnsembleProgressTracker
    Progress tracker for ensemble model training that provides visual
    feedback for split/fold training without verbose output.

Usage Example
-------------
>>> with RayLogManager(
...     mlflow_run_id="run_123",
...     output_dir=Path("./results"),
...     verbose=1,
...     fail_on_upload_error=True,
... ) as manager:
...     # Run HPO or ensemble training with Ray Tune
...     results = tune.run(...)
"""

from __future__ import annotations

import atexit
import gzip
import json
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import mlflow
from ray import tune
from ray.air import session
from ray.tune.callback import Callback
from ray.tune.progress_reporter import ProgressReporter

if TYPE_CHECKING:
    from ray.tune.experiment import Trial
    from ray.tune.result_grid import ResultGrid

logger = logging.getLogger(__name__)


@dataclass
class RayLoggingConfig:
    """Configuration for Ray Tune logging behavior.

    Attributes
    ----------
    enabled : bool
        Whether to enable Ray logging to files. Default: True.
    verbose : int
        Logging verbosity level (0=quiet, 1=standard, 2=debug). Default: 0.
    max_total_logs_gb : float
        Maximum total size (in GB) of logs per experiment. Older logs
        are truncated if this limit is exceeded. Default: 1.0.
    fail_on_upload_error : bool
        If True, raise exception immediately on MLflow upload failure.
        If False, log warning and continue. Default: True.
    """

    enabled: bool = True
    verbose: int = 0
    max_total_logs_gb: float = 1.0
    fail_on_upload_error: bool = True


class RayLogManager:
    """Context manager for Ray Tune logging configuration and artifact upload.

    Configures Ray environment variables to enable logging, collects trial
    logs after experiment completion, compresses them, and uploads to MLflow
    as artifacts. Handles cleanup and interrupt signals gracefully.

    Parameters
    ----------
    mlflow_run_id : str
        The MLflow run ID to associate uploaded artifacts with.
    output_dir : Path
        Directory where Ray Tune saves results and trial logs.
    verbose : int, optional
        Logging verbosity (0=quiet, 1=standard, 2=debug). Default: 0.
    max_total_logs_gb : float, optional
        Max total logs size in GB per experiment. Default: 1.0.
    fail_on_upload_error : bool, optional
        If True, raise on upload failure; else log warning. Default: True.

    Attributes
    ----------
    log_dir : Path
        Directory containing collected logs (created on __enter__).
    logs_archive_path : Path or None
        Path to compressed logs archive after collection.

    Notes
    -----
    Ray workers run in separate processes, so sys.stdout/stderr redirection
    in the main process will NOT capture worker output. This manager uses
    Ray's built-in logging infrastructure via environment variables:
    - RAY_LOG_TO_DRIVER: Enable per-worker log output
    - RAY_LOGGING_LEVEL: Set logging level for Ray components

    After experiment completion, logs are collected from trial directories,
    compressed (gzip), and uploaded to MLflow. Total size is limited to
    prevent excessive artifact storage.

    Example
    -------
    >>> with RayLogManager(
    ...     mlflow_run_id="abc123",
    ...     output_dir=Path("./results"),
    ...     verbose=1,
    ... ) as manager:
    ...     results = tune.run(...)
    ...     # Logs automatically collected and uploaded
    """

    def __init__(
        self,
        mlflow_run_id: str,
        output_dir: Path,
        verbose: int = 0,
        max_total_logs_gb: float = 1.0,
        fail_on_upload_error: bool = True,
    ):
        """Initialize the Ray logging manager."""
        self.mlflow_run_id = mlflow_run_id
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.max_total_logs_gb = max_total_logs_gb
        self.fail_on_upload_error = fail_on_upload_error

        self.log_dir = self.output_dir / "logs"
        self.logs_archive_path: Optional[Path] = None

        self._original_env = {}
        self._signal_handlers = {}

    def __enter__(self) -> RayLogManager:
        """Configure Ray logging and set up cleanup handlers."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ray logging enabled: output_dir={self.output_dir}, verbose={self.verbose}")

        # Configure Ray environment variables for logging
        self._original_env["RAY_LOG_TO_DRIVER"] = os.environ.get("RAY_LOG_TO_DRIVER")
        self._original_env["RAY_LOGGING_LEVEL"] = os.environ.get("RAY_LOGGING_LEVEL")

        if self.verbose >= 1:
            os.environ["RAY_LOG_TO_DRIVER"] = "1"
            os.environ["RAY_LOGGING_LEVEL"] = "info"
        else:
            os.environ["RAY_LOG_TO_DRIVER"] = "0"
            os.environ["RAY_LOGGING_LEVEL"] = "warning"

        # Register signal handlers for graceful cleanup on interrupt
        def _signal_handler(signum: int, frame: Any) -> None:
            logger.warning(f"Received signal {signum}, collecting and uploading logs...")
            self._collect_and_upload_logs()
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        self._signal_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, _signal_handler)
        self._signal_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, _signal_handler)

        # Register cleanup on normal exit
        atexit.register(self._cleanup)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Collect logs, upload to MLflow, and restore original state."""
        try:
            self._collect_and_upload_logs()
        finally:
            self._cleanup()

    def _collect_and_upload_logs(self) -> None:
        """Collect logs from trial directories, compress, and upload to MLflow."""
        if not self.mlflow_run_id:
            logger.warning("No MLflow run ID; skipping log upload")
            return

        try:
            # Collect all log files from trial directories
            log_files = self._collect_trial_logs()

            if not log_files:
                logger.info("No trial logs found to upload")
                return

            # Create compressed archive
            self.logs_archive_path = self._compress_logs(log_files)

            # Upload to MLflow with size enforcement
            self._upload_logs_to_mlflow(self.logs_archive_path)
            logger.info(f"Logs uploaded to MLflow: {self.logs_archive_path.name}")

        except Exception as e:
            msg = f"Error uploading logs to MLflow: {e}"
            if self.fail_on_upload_error:
                logger.error(msg)
                raise
            else:
                logger.warning(msg)

    def _collect_trial_logs(self) -> list[Path]:
        """Collect log files from all trial directories.

        Returns
        -------
        list[Path]
            List of log file paths found in trial directories.
        """
        log_files = []

        # Ray Tune creates trial directories with pattern: trialname_XXXXX/
        trial_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name != "logs"]

        for trial_dir in trial_dirs:
            # Look for common log file patterns
            for pattern in ["*.log", "logs/*.log", "*/*.log"]:
                log_files.extend(trial_dir.glob(pattern))

            # Also look for Ray actor logs
            ray_logs_dir = trial_dir / "logs"
            if ray_logs_dir.exists():
                log_files.extend(ray_logs_dir.glob("*.log"))

        logger.info(f"Collected {len(log_files)} log files from trials")
        return log_files

    def _compress_logs(self, log_files: list[Path]) -> Path:
        """Compress collected log files into a gzip archive.

        Parameters
        ----------
        log_files : list[Path]
            Log files to compress.

        Returns
        -------
        Path
            Path to created archive.

        Notes
        -----
        Archive is stored in the logs directory with timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"ray_logs_{timestamp}.tar.gz"
        archive_path = self.log_dir / archive_name

        try:
            import tarfile

            with tarfile.open(archive_path, "w:gz") as tar:
                for log_file in log_files:
                    try:
                        # Add with relative path to keep structure clean
                        tar.add(log_file, arcname=log_file.relative_to(self.output_dir))
                    except Exception as e:
                        logger.warning(f"Failed to add {log_file} to archive: {e}")

            # Enforce size limit
            archive_size_gb = archive_path.stat().st_size / (1024**3)
            if archive_size_gb > self.max_total_logs_gb:
                logger.warning(
                    f"Archive size {archive_size_gb:.2f} GB exceeds max {self.max_total_logs_gb} GB. "
                    "Removing oldest logs..."
                )
                self._truncate_logs(self.max_total_logs_gb)

            logger.info(f"Compressed logs to {archive_path.name} ({archive_size_gb:.2f} GB)")
            return archive_path

        except Exception as e:
            logger.error(f"Failed to compress logs: {e}")
            raise

    def _truncate_logs(self, max_size_gb: float) -> None:
        """Remove oldest logs to enforce size limit.

        Parameters
        ----------
        max_size_gb : float
            Maximum total size in GB.
        """
        max_size_bytes = max_size_gb * (1024**3)
        total_size = 0
        log_files = sorted(self.log_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)

        # Remove oldest files until under limit
        for log_file in log_files:
            size = log_file.stat().st_size
            total_size += size
            if total_size > max_size_bytes:
                logger.info(f"Removing old log archive: {log_file.name}")
                log_file.unlink()

    def _upload_logs_to_mlflow(self, archive_path: Path) -> None:
        """Upload compressed logs to MLflow as artifact.

        Parameters
        ----------
        archive_path : Path
            Path to compressed logs archive.

        Raises
        ------
        Exception
            If upload fails and fail_on_upload_error is True.
        """
        try:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                mlflow.log_artifact(str(archive_path), artifact_path="logs")
        except Exception as e:
            if self.fail_on_upload_error:
                raise
            else:
                logger.warning(f"Failed to upload logs to MLflow: {e}")

    def _cleanup(self) -> None:
        """Restore original environment variables and signal handlers."""
        # Restore environment
        for key, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

        # Restore signal handlers
        for signum, original_handler in self._signal_handlers.items():
            signal.signal(signum, original_handler)

        logger.debug("Ray logging manager cleanup complete")


class QuietProgressReporter(ProgressReporter):
    """Minimal progress reporter for Ray Tune with clean terminal output.

    Only displays completion summary instead of per-trial details,
    significantly reducing terminal verbosity during HPO and ensemble training.

    Parameters
    ----------
    metric_columns : list[str], optional
        Columns to display in summary. Default: None (uses Ray defaults).

    Notes
    -----
    This reporter is optimized for production runs where the main interest
    is completion status and final results, not per-trial intermediate metrics.

    Example
    -------
    >>> tuner = tune.Tuner(
    ...     train_fn,
    ...     tune_config=tune.TuneConfig(progress_reporter=QuietProgressReporter()),
    ... )
    >>> results = tuner.fit()
    """

    def __init__(self, metric_columns: Optional[list[str]] = None):
        """Initialize the quiet progress reporter."""
        # ProgressReporter is an abstract class without __init__ parameters
        # Store metric_columns for potential future use
        self._metric_columns = metric_columns or []
        self._last_update_time = time.time()
        self._update_interval = 5.0  # Only update every 5 seconds

    def report_progress(self, trials: Any, done: bool = False) -> None:
        """Report progress with minimal output.

        Parameters
        ----------
        trials : Any
            Ray Tune trial list.
        done : bool
            Whether the experiment is complete.
        """
        now = time.time()

        # Only update periodically to reduce noise
        if not done and (now - self._last_update_time) < self._update_interval:
            return

        self._last_update_time = now

        # Count trial statuses
        total = len(trials)
        completed = sum(1 for t in trials if t.status == "TERMINATED")
        running = sum(1 for t in trials if t.status == "RUNNING")
        errored = sum(1 for t in trials if t.status == "ERROR")

        # Display only summary
        status = f"Trials: {completed}/{total} completed"
        if running > 0:
            status += f", {running} running"
        if errored > 0:
            status += f", {errored} errored"

        if done:
            status += " [DONE]"

        logger.info(status)

    def should_report(self, trials: Any, done: bool = False) -> bool:
        """Determine if progress should be reported.

        Parameters
        ----------
        trials : Any
            Ray Tune trial list.
        done : bool
            Whether experiment is complete.

        Returns
        -------
        bool
            True if should report (only on completion or first run).
        """
        return done or (time.time() - self._last_update_time) >= self._update_interval


class LogArtifactCallback(Callback):
    """Ray Tune callback to upload trial logs to MLflow after experiment.

    Collects logs from all completed trials and uploads them as a
    compressed artifact to MLflow when the experiment finishes.

    Parameters
    ----------
    mlflow_run_id : str
        The MLflow run ID to associate uploaded artifacts with.
    output_dir : Path
        Directory containing trial results and logs.
    max_total_logs_gb : float, optional
        Maximum total logs size in GB. Default: 1.0.
    fail_on_upload_error : bool, optional
        If True, raise on upload failure; else log warning. Default: True.

    Notes
    -----
    This callback is called by Ray Tune after experiment completion.
    It is separate from RayLogManager to allow flexible integration
    with different experiment orchestration patterns.

    Example
    -------
    >>> callback = LogArtifactCallback(
    ...     mlflow_run_id="run_123",
    ...     output_dir=Path("./results"),
    ... )
    >>> tuner = tune.Tuner(
    ...     train_fn,
    ...     tune_config=tune.TuneConfig(callbacks=[callback]),
    ... )
    """

    def __init__(
        self,
        mlflow_run_id: str,
        output_dir: Path,
        max_total_logs_gb: float = 1.0,
        fail_on_upload_error: bool = True,
    ):
        """Initialize the log artifact callback."""
        self.mlflow_run_id = mlflow_run_id
        self.output_dir = Path(output_dir)
        self.max_total_logs_gb = max_total_logs_gb
        self.fail_on_upload_error = fail_on_upload_error

    def on_experiment_end(self, algorithm: Any, trials: Any, **info: Any) -> None:
        """Upload collected logs when experiment ends.

        Parameters
        ----------
        algorithm : Any
            Ray Tune algorithm instance.
        trials : Any
            List of all trials in the experiment.
        **info : Any
            Additional context info.
        """
        if not self.mlflow_run_id:
            logger.warning("No MLflow run ID; skipping log upload")
            return

        try:
            # Collect logs from all trials
            log_files = self._collect_trial_logs()

            if not log_files:
                logger.info("No trial logs found to upload")
                return

            # Compress and upload
            archive_path = self._compress_logs(log_files)
            self._upload_to_mlflow(archive_path)

        except Exception as e:
            msg = f"Error in LogArtifactCallback: {e}"
            if self.fail_on_upload_error:
                logger.error(msg)
                raise
            else:
                logger.warning(msg)

    def _collect_trial_logs(self) -> list[Path]:
        """Collect log files from trial directories."""
        log_files = []
        trial_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name != "logs"]

        for trial_dir in trial_dirs:
            log_files.extend(trial_dir.glob("*.log"))
            logs_subdir = trial_dir / "logs"
            if logs_subdir.exists():
                log_files.extend(logs_subdir.glob("*.log"))

        return log_files

    def _compress_logs(self, log_files: list[Path]) -> Path:
        """Compress log files into archive."""
        import tarfile

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"ray_logs_{timestamp}.tar.gz"
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        archive_path = logs_dir / archive_name

        with tarfile.open(archive_path, "w:gz") as tar:
            for log_file in log_files:
                try:
                    tar.add(log_file, arcname=log_file.relative_to(self.output_dir))
                except Exception as e:
                    logger.warning(f"Failed to add {log_file}: {e}")

        return archive_path

    def _upload_to_mlflow(self, archive_path: Path) -> None:
        """Upload archive to MLflow."""
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_artifact(str(archive_path), artifact_path="logs")


class EnsembleProgressTracker:
    """Progress tracker for ensemble model training with clean display.

    Provides visual feedback for split/fold training completion without
    verbose output, updating periodically with estimated time remaining.

    Parameters
    ----------
    total_models : int
        Total number of models to train.
    update_interval : float, optional
        Seconds between progress updates. Default: 10.0.

    Example
    -------
    >>> tracker = EnsembleProgressTracker(total_models=20)
    >>> for i, model in enumerate(models):
    ...     model.fit()
    ...     tracker.update(i + 1)
    """

    def __init__(self, total_models: int, update_interval: float = 10.0):
        """Initialize the ensemble progress tracker."""
        self.total_models = total_models
        self.update_interval = update_interval
        self.completed = 0
        self.start_time = time.time()
        self._last_update_time = time.time()

    def update(self, completed: int) -> None:
        """Update progress with completion count.

        Parameters
        ----------
        completed : int
            Number of models completed so far.
        """
        self.completed = completed
        now = time.time()

        # Only log periodically
        if (now - self._last_update_time) < self.update_interval and completed < self.total_models:
            return

        self._last_update_time = now
        elapsed = now - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0

        if rate > 0:
            remaining = (self.total_models - self.completed) / rate
            eta_str = f", ETA: {remaining:.1f}s"
        else:
            eta_str = ""

        logger.info(
            f"Ensemble training: {self.completed}/{self.total_models} models completed"
            f" ({elapsed:.1f}s elapsed{eta_str})"
        )

    def finish(self) -> None:
        """Log completion message."""
        elapsed = time.time() - self.start_time
        logger.info(f"Ensemble training complete: {self.total_models} models in {elapsed:.1f}s")
