<!-- markdownlint-disable-file -->

# Task Details: Ray Tune Output Logging to MLflow Artifacts

## Research Reference

**Critical Issue**: Ray workers run in separate processes. Redirecting `sys.stdout`/`sys.stderr` in the main process will NOT capture Ray worker output. Must use Ray's native logging infrastructure.

## User Decisions (Finalized)

Based on user feedback, the following design decisions are finalized:

| Decision | Value | Notes |
|----------|-------|-------|
| Default verbosity | `0` | Quiet mode, only progress indicators |
| Max total logs GB | `1` | Per experiment, enforced by truncation |
| CLI flags | Yes | `--logging-verbose N`, `--no-logging` |
| Fail on upload error | `True` | Immediate failure, no retries |
| Testing approach | Progressive | Tests between implementation phases |
| Config updates | Aggressive | ALL 200+ YAML files get logging section |
| Documentation | Rich | Comprehensive guide with troubleshooting |
| Performance benchmarks | Yes | Compression overhead, memory usage |

---

## Phase 1: Core Logging Infrastructure

### Task 1.1: Create RayLogManager Class

Create `src/admet/util/ray_logging.py` with a manager class that configures Ray logging and collects logs post-run.

- **Files**:
  - `src/admet/util/ray_logging.py` - NEW: Ray logging utilities

- **Implementation**:

```python
"""Ray Tune logging utilities for capturing and managing output.

This module provides utilities to configure Ray's logging infrastructure,
redirect verbose output to files, and upload logs as MLflow artifacts.

Key components:
- RayLogManager: Context manager for Ray logging configuration
- QuietProgressReporter: Minimal terminal output for trials
- LogArtifactCallback: Upload logs to MLflow after run completion
"""

from __future__ import annotations

import atexit
import gzip
import logging
import os
import shutil
import signal
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from ray.tune import ResultGrid

logger = logging.getLogger(__name__)


class RayLogManager:
    """Manage Ray Tune logging with file capture and MLflow upload.

    This class configures Ray's logging to write to files instead of flooding
    the terminal, while maintaining minimal progress output. Logs are
    automatically uploaded to MLflow as artifacts.

    Attributes
    ----------
    log_dir : Path
        Directory for log files.
    experiment_name : str
        Name used for log file prefixes.
    mlflow_run_id : str | None
        MLflow run ID for artifact upload.
    compress : bool
        Whether to gzip compress logs before upload.

    Examples
    --------
    >>> with RayLogManager(
    ...     log_dir=Path("output/logs"),
    ...     experiment_name="hpo_run",
    ...     mlflow_run_id=run.info.run_id,
    ... ) as log_manager:
    ...     results = tuner.fit()
    ...     log_manager.set_results(results)
    """

    def __init__(
        self,
        log_dir: Path | str,
        experiment_name: str,
        mlflow_run_id: str | None = None,
        compress: bool = True,
        max_log_size_mb: int = 100,
        max_total_logs_gb: int = 1,
        fail_on_upload_error: bool = True,
        timestamp: str | None = None,
    ) -> None:
        """Initialize the log manager.

        Parameters
        ----------
        log_dir : Path | str
            Directory for log files.
        experiment_name : str
            Name for log file prefix.
        mlflow_run_id : str | None
            MLflow run ID for artifact upload.
        compress : bool
            Compress logs with gzip before upload.
        max_log_size_mb : int
            Maximum individual log file size to upload.
        max_total_logs_gb : int
            Maximum total log size per experiment (default: 1 GB).
        fail_on_upload_error : bool
            Raise exception on upload failure (default: True).
        timestamp : str | None
            Timestamp string; generated if None.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = experiment_name.replace("/", "_").replace(" ", "_")

        self.experiment_name = experiment_name
        self.mlflow_run_id = mlflow_run_id
        self.compress = compress
        self.max_log_size_mb = max_log_size_mb
        self.max_total_logs_gb = max_total_logs_gb
        self.fail_on_upload_error = fail_on_upload_error

        # Log file paths
        self.main_log = self.log_dir / f"{ts}_{safe_name}_main.log"
        self.ray_session_dir: Path | None = None

        # State
        self._original_env: dict[str, str | None] = {}
        self._results: ResultGrid | None = None
        self._cleanup_registered = False

    def __enter__(self) -> "RayLogManager":
        """Configure Ray logging environment."""
        self._configure_ray_logging()
        self._register_cleanup_handlers()
        logger.info("Ray logging configured. Logs: %s", self.log_dir)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore environment and upload logs."""
        self._restore_environment()
        self._collect_and_upload_logs()

    def set_results(self, results: "ResultGrid") -> None:
        """Store results for log collection."""
        self._results = results

    def _configure_ray_logging(self) -> None:
        """Set Ray environment variables for quiet logging."""
        env_settings = {
            # Reduce Ray Tune verbosity
            "RAY_AIR_NEW_OUTPUT": "0",  # Disable new output format (cleaner)
            "TUNE_DISABLE_STRICT_METRIC_CHECKING": "1",
            "TUNE_WARN_THRESHOLD_S": "60",  # Only warn if >60s
            "TUNE_RESULT_BUFFER_LENGTH": "20",
            "TUNE_RESULT_BUFFER_MIN_TIME_S": "30",
            # Ray core logging
            "RAY_LOG_TO_DRIVER": "0",  # Don't forward worker logs to driver
            "RAY_LOGGING_LEVEL": "WARNING",
        }

        for key, value in env_settings.items():
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_environment(self) -> None:
        """Restore original environment variables."""
        for key, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value

    def _register_cleanup_handlers(self) -> None:
        """Register signal and exit handlers for graceful cleanup."""
        if self._cleanup_registered:
            return

        def cleanup_handler(signum: int | None = None, frame: Any = None) -> None:
            logger.info("Cleanup triggered (signal=%s)", signum)
            self._collect_and_upload_logs()

        # Register for graceful shutdown
        atexit.register(cleanup_handler)

        # Handle common interrupt signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, cleanup_handler)
            except (OSError, ValueError):
                pass  # May fail in non-main thread

        self._cleanup_registered = True

    def _collect_and_upload_logs(self) -> None:
        """Collect Ray logs and upload to MLflow."""
        if not self.mlflow_run_id:
            logger.debug("No MLflow run ID; skipping log upload")
            return

        try:
            # Collect trial logs from Ray session directory
            collected_logs = self._collect_trial_logs()

            # Enforce max total size limit (default 1 GB)
            collected_logs = self._enforce_size_limit(collected_logs)

            # Compress if requested
            if self.compress:
                collected_logs = self._compress_logs(collected_logs)

            # Upload to MLflow
            self._upload_to_mlflow(collected_logs)

        except Exception as e:
            if self.fail_on_upload_error:
                raise RuntimeError(f"Failed to upload logs to MLflow: {e}") from e
            logger.warning("Failed to upload logs to MLflow: %s", e)

    def _enforce_size_limit(self, logs: list[Path]) -> list[Path]:
        """Enforce maximum total log size limit.

        Parameters
        ----------
        logs : list[Path]
            List of log files to check.

        Returns
        -------
        list[Path]
            Log files that fit within the limit.
        """
        max_bytes = self.max_total_logs_gb * 1024 * 1024 * 1024
        total_bytes = 0
        selected_logs: list[Path] = []

        # Sort by size (smallest first) to maximize number of files
        sorted_logs = sorted(logs, key=lambda p: p.stat().st_size)

        for log_path in sorted_logs:
            size = log_path.stat().st_size
            if total_bytes + size <= max_bytes:
                selected_logs.append(log_path)
                total_bytes += size
            else:
                logger.warning(
                    "Skipping %s (%.2f MB) - would exceed %.1f GB limit",
                    log_path.name,
                    size / (1024 * 1024),
                    self.max_total_logs_gb,
                )

        if len(selected_logs) < len(logs):
            logger.warning(
                "Log size limit enforced: %d/%d files (%.2f/%.2f GB)",
                len(selected_logs),
                len(logs),
                total_bytes / (1024 * 1024 * 1024),
                self.max_total_logs_gb,
            )

        return selected_logs

    def _collect_trial_logs(self) -> list[Path]:
        """Collect log files from Ray trial directories."""
        logs: list[Path] = []

        # Main application log
        if self.main_log.exists():
            logs.append(self.main_log)

        # Find Ray session logs if available
        if self._results is not None:
            for result in self._results:
                trial_dir = Path(result.path) if result.path else None
                if trial_dir and trial_dir.exists():
                    # Collect stdout/stderr logs
                    for log_file in trial_dir.glob("*.log"):
                        if log_file.stat().st_size < self.max_log_size_mb * 1024 * 1024:
                            logs.append(log_file)

        return logs

    def _compress_logs(self, logs: list[Path]) -> list[Path]:
        """Compress log files with gzip."""
        compressed: list[Path] = []

        for log_path in logs:
            gz_path = log_path.with_suffix(log_path.suffix + ".gz")
            try:
                with open(log_path, "rb") as f_in:
                    with gzip.open(gz_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                compressed.append(gz_path)
            except Exception as e:
                logger.warning("Failed to compress %s: %s", log_path, e)
                compressed.append(log_path)  # Use uncompressed

        return compressed

    def _upload_to_mlflow(self, logs: list[Path]) -> None:
        """Upload log files as MLflow artifacts."""
        if not logs:
            return

        try:
            with mlflow.start_run(run_id=self.mlflow_run_id):
                for log_path in logs:
                    mlflow.log_artifact(str(log_path), artifact_path="logs")
                logger.info("Uploaded %d log files to MLflow", len(logs))
        except Exception as e:
            logger.warning("MLflow upload failed: %s", e)
```

- **Success**:
  - Context manager properly configures Ray environment
  - Logs collected from trial directories
  - Compression works correctly
  - MLflow upload succeeds
  - Graceful cleanup on interrupts

- **Dependencies**: None (first task)

---

### Task 1.2: Create QuietProgressReporter

Add a custom `ProgressReporter` that provides minimal terminal output.

- **Files**:
  - `src/admet/util/ray_logging.py` - ADD: QuietProgressReporter class

- **Implementation**:

```python
from ray.tune import CLIReporter
from ray.tune.experiment import Trial


class QuietProgressReporter(CLIReporter):
    """Minimal progress reporter for clean terminal output.

    Only prints trial completion summary and errors, not per-iteration updates.

    Parameters
    ----------
    metric : str
        Metric to display in progress.
    mode : str
        'min' or 'max' for metric optimization direction.
    """

    def __init__(
        self,
        metric: str = "val_loss",
        mode: str = "min",
        max_report_frequency: int = 30,
    ) -> None:
        super().__init__(
            metric_columns=[metric],
            max_report_frequency=max_report_frequency,
            print_intermediate_tables=False,  # Don't print intermediate tables
        )
        self.metric = metric
        self.mode = mode
        self._completed = 0
        self._total = 0
        self._best_value: float | None = None

    def setup(
        self,
        start_time: float | None = None,
        total_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize reporter with trial count."""
        super().setup(start_time=start_time, total_samples=total_samples, **kwargs)
        self._total = total_samples or 0
        print(f"\n🔬 HPO: {self._total} trials | metric: {self.metric} ({self.mode})")
        print("=" * 50)

    def report(self, trials: list[Trial], done: bool, *args: Any, **kwargs: Any) -> None:
        """Report trial progress with minimal output."""
        completed = sum(1 for t in trials if t.status == Trial.TERMINATED)

        # Only print on new completions
        if completed > self._completed:
            self._completed = completed

            # Find best trial
            best_trial = None
            for t in trials:
                if t.status == Trial.TERMINATED and t.last_result:
                    value = t.last_result.get(self.metric)
                    if value is not None:
                        if self._best_value is None:
                            self._best_value = value
                            best_trial = t
                        elif (self.mode == "min" and value < self._best_value) or \
                             (self.mode == "max" and value > self._best_value):
                            self._best_value = value
                            best_trial = t

            # Print progress line
            best_str = f" | best {self.metric}: {self._best_value:.4f}" if self._best_value else ""
            print(f"[{self._completed}/{self._total}] Trials completed{best_str}")

        if done:
            print("=" * 50)
            print(f"✓ HPO complete. Best {self.metric}: {self._best_value:.4f}")

    def should_report(self, trials: list[Trial], done: bool = False) -> bool:
        """Always report on completion or errors."""
        if done:
            return True
        completed = sum(1 for t in trials if t.status == Trial.TERMINATED)
        return completed > self._completed
```

- **Success**:
  - Terminal output limited to completion summaries
  - Best metric tracked and displayed
  - Integrates with Ray Tune via `progress_reporter` parameter

- **Dependencies**: Task 1.1

---

### Task 1.3: Create LogArtifactCallback

Add a Ray Tune callback that uploads logs after experiment completion.

- **Files**:
  - `src/admet/util/ray_logging.py` - ADD: LogArtifactCallback class

- **Implementation**:

```python
from ray.tune.callback import Callback
from ray.tune.experiment import Trial


class LogArtifactCallback(Callback):
    """Ray Tune callback to upload trial logs to MLflow.

    Collects logs from completed trials and uploads them as artifacts
    to the parent MLflow run.

    Parameters
    ----------
    mlflow_run_id : str
        Parent MLflow run ID.
    log_dir : Path
        Directory for collected logs.
    compress : bool
        Whether to compress logs before upload.
    """

    def __init__(
        self,
        mlflow_run_id: str,
        log_dir: Path,
        compress: bool = True,
    ) -> None:
        self.mlflow_run_id = mlflow_run_id
        self.log_dir = Path(log_dir)
        self.compress = compress
        self._trial_logs: dict[str, Path] = {}

    def on_trial_complete(
        self,
        iteration: int,
        trials: list[Trial],
        trial: Trial,
        **info: Any,
    ) -> None:
        """Record trial log location on completion."""
        if trial.logdir:
            self._trial_logs[trial.trial_id] = Path(trial.logdir)

    def on_experiment_end(self, trials: list[Trial], **info: Any) -> None:
        """Upload all collected logs to MLflow."""
        if not self.mlflow_run_id:
            return

        try:
            logs_to_upload: list[Path] = []

            for trial_id, logdir in self._trial_logs.items():
                for log_file in logdir.glob("*.log"):
                    # Skip very large files
                    if log_file.stat().st_size > 50 * 1024 * 1024:  # 50MB
                        continue
                    logs_to_upload.append(log_file)

            if not logs_to_upload:
                return

            # Create summary log
            summary_path = self.log_dir / "trial_summary.log"
            with open(summary_path, "w") as f:
                f.write(f"HPO Summary: {len(trials)} trials\n")
                f.write("=" * 50 + "\n")
                for t in trials:
                    status = t.status
                    result = t.last_result or {}
                    f.write(f"Trial {t.trial_id}: {status}\n")
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and not k.startswith("_"):
                            f.write(f"  {k}: {v}\n")
                    f.write("\n")
            logs_to_upload.append(summary_path)

            # Upload to MLflow
            with mlflow.start_run(run_id=self.mlflow_run_id):
                for log_path in logs_to_upload:
                    mlflow.log_artifact(str(log_path), artifact_path="logs/trials")

            logger.info("Uploaded %d trial logs to MLflow", len(logs_to_upload))

        except Exception as e:
            logger.warning("Failed to upload trial logs: %s", e)
```

- **Success**:
  - Collects logs from all completed trials
  - Creates summary log file
  - Uploads to MLflow on experiment end

- **Dependencies**: Task 1.1

---

## Phase 2: Configuration Schema

### Task 2.1: Add RayLoggingConfig Dataclass

Add logging configuration to `src/admet/model/config.py`.

- **Files**:
  - `src/admet/model/config.py` - ADD: RayLoggingConfig dataclass

- **Implementation**:

```python
@dataclass
class RayLoggingConfig:
    """Configuration for Ray Tune logging behavior.

    Controls verbosity, log file locations, and MLflow artifact upload.

    Attributes
    ----------
    enabled : bool
        Enable log file capture and MLflow upload.
    log_subdir : str
        Subdirectory under output_dir for logs.
    verbose : int
        Ray Tune verbosity level (0=quiet, 1=status, 2=brief, 3=detailed).
    compress : bool
        Compress logs with gzip before MLflow upload.
    upload_to_mlflow : bool
        Upload logs as MLflow artifacts.
    max_log_size_mb : int
        Maximum individual log file size to upload.
    max_total_logs_gb : int
        Maximum total log size per experiment.
    fail_on_upload_error : bool
        Raise exception on upload failure.
    """

    enabled: bool = True
    log_subdir: str = "logs"
    verbose: int = 0  # Quiet mode by default (user decision)
    compress: bool = True
    upload_to_mlflow: bool = True
    max_log_size_mb: int = 100
    max_total_logs_gb: int = 1  # 1 GB limit per experiment (user decision)
    fail_on_upload_error: bool = True  # Fail-fast (user decision)
```

- **Location**: Add after existing config dataclasses (around line 400)

- **Success**:
  - Dataclass validates correctly
  - Default values are sensible for production

- **Dependencies**: None

---

### Task 2.2: Add logging Field to HPOConfig and EnsembleConfig

Update `HPOConfig` and `EnsembleConfig` to include logging configuration.

- **Files**:
  - `src/admet/model/config.py` - MODIFY: Add logging field
  - `src/admet/model/chemprop/hpo_config.py` - MODIFY: Add logging field

- **Changes to HPOConfig** (in `hpo_config.py`):

```python
from admet.model.config import RayLoggingConfig

@dataclass
class HPOConfig:
    # ... existing fields ...
    logging: RayLoggingConfig = field(default_factory=RayLoggingConfig)
```

- **Changes to EnsembleConfig** (in `config.py`):

```python
@dataclass
class EnsembleConfig:
    # ... existing fields ...
    logging: RayLoggingConfig = field(default_factory=RayLoggingConfig)
```

- **Success**:
  - Configs load correctly with logging section
  - OmegaConf merging works

- **Dependencies**: Task 2.1

---

## Phase 3: CLI Integration

### Task 3.1: Add `--logging-verbose` CLI Flag

Add CLI flag to override logging verbosity from command line.

- **Files**:
  - `src/admet/cli/model.py` - MODIFY: Add logging flags

- **Implementation**:

```python
from typing import Annotated

# In HPO command:
@model_app.command("hpo")
def hpo(
    config: str = typer.Option(..., "-c", "--config", help="Config file path"),
    logging_verbose: Annotated[int | None, typer.Option(
        "--logging-verbose",
        help="Override logging verbosity (0=quiet, 1=status, 2=brief, 3=detailed)",
    )] = None,
    no_logging: Annotated[bool, typer.Option(
        "--no-logging",
        help="Disable logging to files and MLflow",
    )] = False,
    # ... existing options ...
) -> None:
    """Run hyperparameter optimization."""
    # Load config
    cfg = OmegaConf.load(config)

    # Apply CLI overrides
    if no_logging:
        cfg.logging.enabled = False
    if logging_verbose is not None:
        cfg.logging.verbose = logging_verbose

    # ... rest of command ...
```

- **Success**:
  - `admet model hpo -c config.yaml --logging-verbose 1` works
  - `admet model hpo -c config.yaml --no-logging` works
  - CLI help shows both flags

- **Dependencies**: Phase 2

---

### Task 3.2: Add Flags to Ensemble Command

Add same flags to ensemble training command.

- **Files**:
  - `src/admet/cli/model.py` - MODIFY: Add flags to ensemble command

- **Implementation**:

```python
# In ensemble command:
@model_app.command("ensemble")
def ensemble(
    config: str = typer.Option(..., "-c", "--config", help="Config file path"),
    logging_verbose: Annotated[int | None, typer.Option(
        "--logging-verbose",
        help="Override logging verbosity (0=quiet, 1=status, 2=brief, 3=detailed)",
    )] = None,
    no_logging: Annotated[bool, typer.Option(
        "--no-logging",
        help="Disable logging to files and MLflow",
    )] = False,
    # ... existing options ...
) -> None:
    """Train ensemble of models."""
    # Same override pattern as HPO command
    # ...
```

- **Success**:
  - Both commands have consistent logging flags
  - Flags documented in CLI help

- **Dependencies**: Task 3.1

---

## Phase 4: HPO Integration

### Task 4.1: Update ChempropHPO.run()

Integrate `RayLogManager` and `QuietProgressReporter` into Chemprop HPO.

- **Files**:
  - `src/admet/model/chemprop/hpo.py` - MODIFY: Add logging integration

- **Changes**:

```python
# Add imports at top
from admet.util.ray_logging import (
    LogArtifactCallback,
    QuietProgressReporter,
    RayLogManager,
)

# In ChempropHPO.run() method:
def run(self) -> tune.ResultGrid:
    """Run hyperparameter optimization."""
    self._setup_mlflow()
    # ... existing setup ...

    # Configure logging
    log_dir = Path(self.config.output_dir) / self.config.logging.log_subdir / "hpo"
    log_manager = RayLogManager(
        log_dir=log_dir,
        experiment_name=self.config.experiment_name,
        mlflow_run_id=self._mlflow_run_id,
        compress=self.config.logging.compress,
    ) if self.config.logging.enabled else None

    # Build callbacks list
    callbacks = [mlflow_callback]
    if self.config.logging.enabled and self._mlflow_run_id:
        callbacks.append(LogArtifactCallback(
            mlflow_run_id=self._mlflow_run_id,
            log_dir=log_dir,
            compress=self.config.logging.compress,
        ))

    # Build progress reporter
    progress_reporter = QuietProgressReporter(
        metric=self.config.asha.metric,
        mode=self.config.asha.mode,
    ) if self.config.logging.verbose == 0 else None

    # Configure RunConfig with quiet verbosity
    run_config = tune.RunConfig(
        name=self.config.experiment_name,
        storage_path=storage_path,
        verbose=self.config.logging.verbose,
        callbacks=callbacks,
        progress_reporter=progress_reporter,
        # ... existing config ...
    )

    # Use context manager for logging
    context = log_manager if log_manager else contextlib.nullcontext()
    with context:
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        self.results = tuner.fit()

        if log_manager:
            log_manager.set_results(self.results)

    return self.results
```

- **Success**:
  - HPO runs with minimal terminal output
  - Logs captured and uploaded to MLflow
  - Graceful handling of interrupts

- **Dependencies**: Phase 1, Phase 2

---

### Task 3.2: Update ChemeleonHPO.run()

Apply same changes to Chemeleon HPO module.

- **Files**:
  - `src/admet/model/chemeleon/hpo.py` - MODIFY: Add logging integration

- **Changes**: Same pattern as Task 3.1

- **Success**:
  - Chemeleon HPO uses same logging infrastructure
  - Consistent behavior across model types

- **Dependencies**: Task 3.1

---

## Phase 4: Ensemble Integration

### Task 4.1: Update ModelEnsemble.train_all()

Add logging support to ensemble training.

- **Files**:
  - `src/admet/model/chemprop/ensemble.py` - MODIFY: Add logging integration

- **Changes**:

```python
# Add import
from admet.util.ray_logging import RayLogManager

def train_all(self) -> None:
    """Train all models in the ensemble."""
    if not self.split_fold_infos:
        self.discover_splits_and_folds()

    # Configure logging
    log_dir = Path(self.config.data.data_dir).parent / "logs" / "ensemble"
    parent_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    log_manager = RayLogManager(
        log_dir=log_dir,
        experiment_name=getattr(self.config.mlflow, "experiment_name", "ensemble"),
        mlflow_run_id=parent_run_id,
        compress=getattr(self.config.logging, "compress", True),
    ) if getattr(self.config, "logging", None) and self.config.logging.enabled else None

    context = log_manager if log_manager else contextlib.nullcontext()

    with context:
        # Print progress header
        total_models = len(self.split_fold_infos)
        print(f"\n🚀 Ensemble Training: {total_models} models")
        print("=" * 50)

        # ... existing Ray remote task submission ...

        # Track progress
        completed = 0
        for result in all_results:
            completed += 1
            print(f"[{completed}/{total_models}] Models trained")
            # ... existing result processing ...

        print("=" * 50)
        print(f"✓ Ensemble training complete. {total_models} models trained.")
```

- **Success**:
  - Ensemble training logs captured
  - Progress displayed cleanly
  - Logs uploaded to MLflow

- **Dependencies**: Phase 1, Phase 2

---

### Task 4.2: Add Progress Reporting for Ensemble

Add visual progress indicator for ensemble training.

- **Files**:
  - `src/admet/util/ray_logging.py` - ADD: EnsembleProgressTracker class

- **Implementation**:

```python
class EnsembleProgressTracker:
    """Progress tracker for ensemble model training.

    Provides clean terminal output for parallel model training.
    """

    def __init__(self, total_models: int, experiment_name: str = "ensemble") -> None:
        self.total = total_models
        self.completed = 0
        self.failed = 0
        self.experiment_name = experiment_name
        print(f"\n🚀 {experiment_name}: {total_models} models")
        print("=" * 50)

    def update(self, success: bool = True, model_key: str = "") -> None:
        """Update progress with model completion."""
        if success:
            self.completed += 1
        else:
            self.failed += 1
        total_done = self.completed + self.failed
        status = "✓" if success else "✗"
        print(f"[{total_done}/{self.total}] {status} {model_key}")

    def finish(self) -> None:
        """Print completion summary."""
        print("=" * 50)
        if self.failed == 0:
            print(f"✓ Complete: {self.completed}/{self.total} models trained")
        else:
            print(f"⚠ Complete: {self.completed} succeeded, {self.failed} failed")
```

- **Success**:
  - Clean progress display
  - Handles failures gracefully

- **Dependencies**: Task 4.1

---

## Phase 5: Testing

### Task 5.1: Create Unit Tests

Create `tests/test_ray_logging.py` with comprehensive unit tests.

- **Files**:
  - `tests/test_ray_logging.py` - NEW: Unit tests

- **Implementation**:

```python
"""Tests for Ray Tune logging utilities."""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from admet.util.ray_logging import (
    EnsembleProgressTracker,
    LogArtifactCallback,
    QuietProgressReporter,
    RayLogManager,
)


class TestRayLogManager:
    """Tests for RayLogManager class."""

    def test_init_creates_log_directory(self) -> None:
        """Test log directory is created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs" / "nested"
            manager = RayLogManager(
                log_dir=log_dir,
                experiment_name="test_exp",
            )
            assert log_dir.exists()

    def test_context_manager_sets_environment(self) -> None:
        """Test environment variables are set in context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_value = os.environ.get("RAY_LOG_TO_DRIVER")

            with RayLogManager(
                log_dir=tmpdir,
                experiment_name="test",
            ):
                assert os.environ.get("RAY_LOG_TO_DRIVER") == "0"

            # Restored after exit
            assert os.environ.get("RAY_LOG_TO_DRIVER") == original_value

    def test_compress_logs_creates_gzip(self) -> None:
        """Test log compression creates valid gzip files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"
            log_file.write_text("test content\n" * 100)

            manager = RayLogManager(
                log_dir=log_dir,
                experiment_name="test",
            )
            compressed = manager._compress_logs([log_file])

            assert len(compressed) == 1
            assert compressed[0].suffix == ".gz"

            # Verify content
            with gzip.open(compressed[0], "rt") as f:
                content = f.read()
            assert "test content" in content

    @patch("admet.util.ray_logging.mlflow")
    def test_upload_to_mlflow_called(self, mock_mlflow: MagicMock) -> None:
        """Test MLflow upload is called with correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            log_file = log_dir / "test.log"
            log_file.write_text("test content")

            manager = RayLogManager(
                log_dir=log_dir,
                experiment_name="test",
                mlflow_run_id="test_run_123",
                compress=False,
            )
            manager._upload_to_mlflow([log_file])

            mock_mlflow.start_run.assert_called_once_with(run_id="test_run_123")
            mock_mlflow.log_artifact.assert_called()


class TestQuietProgressReporter:
    """Tests for QuietProgressReporter class."""

    def test_setup_prints_header(self, capsys: pytest.CaptureFixture) -> None:
        """Test setup prints experiment header."""
        reporter = QuietProgressReporter(metric="val_loss", mode="min")
        reporter.setup(total_samples=50)

        captured = capsys.readouterr()
        assert "50 trials" in captured.out
        assert "val_loss" in captured.out

    def test_tracks_best_metric(self) -> None:
        """Test best metric is tracked correctly."""
        reporter = QuietProgressReporter(metric="val_loss", mode="min")
        reporter.setup(total_samples=3)

        # Simulate trial results
        mock_trials = []
        for i, loss in enumerate([0.5, 0.3, 0.4]):
            trial = MagicMock()
            trial.status = "TERMINATED"
            trial.last_result = {"val_loss": loss}
            mock_trials.append(trial)

            reporter.report(mock_trials[:i+1], done=(i == 2))

        assert reporter._best_value == 0.3  # Minimum


class TestLogArtifactCallback:
    """Tests for LogArtifactCallback class."""

    def test_on_trial_complete_records_logdir(self) -> None:
        """Test trial log directories are recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = LogArtifactCallback(
                mlflow_run_id="test_run",
                log_dir=Path(tmpdir),
            )

            mock_trial = MagicMock()
            mock_trial.trial_id = "trial_001"
            mock_trial.logdir = "/path/to/trial"

            callback.on_trial_complete(
                iteration=1,
                trials=[mock_trial],
                trial=mock_trial,
            )

            assert "trial_001" in callback._trial_logs


class TestEnsembleProgressTracker:
    """Tests for EnsembleProgressTracker class."""

    def test_progress_updates(self, capsys: pytest.CaptureFixture) -> None:
        """Test progress updates display correctly."""
        tracker = EnsembleProgressTracker(total_models=5, experiment_name="test")

        for i in range(5):
            tracker.update(success=True, model_key=f"model_{i}")

        tracker.finish()

        captured = capsys.readouterr()
        assert "[5/5]" in captured.out
        assert "Complete" in captured.out

    def test_handles_failures(self, capsys: pytest.CaptureFixture) -> None:
        """Test failure tracking displays warning."""
        tracker = EnsembleProgressTracker(total_models=3)
        tracker.update(success=True)
        tracker.update(success=False)
        tracker.update(success=True)
        tracker.finish()

        captured = capsys.readouterr()
        assert "2 succeeded" in captured.out
        assert "1 failed" in captured.out
```

- **Success**:
  - All unit tests pass
  - Coverage for main functionality

- **Dependencies**: Phase 1

---

### Task 5.2: Create Integration Test

Add integration test that runs a small HPO with logging enabled.

- **Files**:
  - `tests/test_ray_logging.py` - ADD: Integration test

- **Implementation**:

```python
@pytest.mark.slow
@pytest.mark.no_mlflow_runs
class TestRayLoggingIntegration:
    """Integration tests for Ray logging with HPO."""

    def test_hpo_with_logging(self, tmp_path: Path) -> None:
        """Test HPO run with logging enabled produces log files."""
        from ray import tune

        # Simple trainable
        def dummy_trainable(config: dict) -> None:
            for i in range(3):
                tune.report({"val_loss": config["lr"] * (i + 1)})

        log_dir = tmp_path / "logs"

        with RayLogManager(
            log_dir=log_dir,
            experiment_name="test_hpo",
        ) as manager:
            tuner = tune.Tuner(
                dummy_trainable,
                param_space={"lr": tune.uniform(0.001, 0.1)},
                tune_config=tune.TuneConfig(num_samples=2),
                run_config=tune.RunConfig(
                    verbose=0,
                    progress_reporter=QuietProgressReporter(metric="val_loss"),
                ),
            )
            results = tuner.fit()
            manager.set_results(results)

        # Verify logs were created
        assert log_dir.exists()
        assert len(list(log_dir.glob("*.log"))) > 0
```

- **Success**:
  - Integration test passes
  - Log files created and contain expected content

- **Dependencies**: Task 5.1

---

## Phase 6: Configuration Updates

All YAML files in `configs/` directory will be updated to include the `logging` section.

### Strategy

Rather than manually editing 100+ files, use a systematic approach:

1. Identify all YAML files: `find configs/ -name "*.yaml"`
2. For each file: add `logging:` section after the `mlflow:` section (or before if no MLflow)
3. Use consistent formatting

The logging section to add to ALL files:

```yaml
# Ray Tune and ensemble logging configuration
logging:
  enabled: true
  log_subdir: logs
  verbose: 0
  compress: true
  upload_to_mlflow: true
  max_log_size_mb: 100
  max_total_logs_gb: 1
  fail_on_upload_error: true
```

### File Locations and Counts

| Category | Location | Count | Notes |
|----------|----------|-------|-------|
| Experiments | `0-experiment/` | 4 | Single model training examples |
| HPO Single | `1-hpo-single/` | 2 | Individual model HPO |
| HPO Ensemble | `2-hpo-ensemble/` | 100+ | Ensemble HPO runs (chemprop) |
| HPO Production | `3-hpo-production/` | ~100 | Production ensemble configs |
| Classical Models | `4-more-models/` | 5-10 | XGBoost, LightGBM, CatBoost |
| Curriculum | `curriculum/` | 2-3 | Curriculum learning configs |
| Task Affinity | `task-affinity/` | 2-3 | Task affinity grouping configs |

---

### Task 6.1: Update Experiment Configs (`0-experiment/`)

Add logging section to:
- `0-experiment/chemprop.yaml`
- `0-experiment/chemeleon.yaml`
- `0-experiment/ensemble_chemprop_production.yaml`
- `0-experiment/ensemble_chemeleon_production.yaml`
- `0-experiment/chemprop_weight_decay_example.yaml`
- `0-experiment/ensemble_joint_sampling_example.yaml`

**Location**: After `mlflow:` section

**Success**: All 4-6 files have logging section, configs load without error

---

### Task 6.2: Update HPO Single Configs (`1-hpo-single/`)

Add logging section to:
- `1-hpo-single/hpo_chemprop.yaml`
- `1-hpo-single/hpo_chemeleon.yaml`

**Location**: After `ray:` section (or after `mlflow:` if no `ray:`)

**Success**: HPO runs with quiet output

---

### Task 6.3: Update HPO Ensemble Configs (`2-hpo-ensemble/`)

Add logging section to all `ensemble_chemprop_hpo_*.yaml` files (100+ files).

**Approach**: Use Python script to batch update:

```python
import yaml
from pathlib import Path

configs_dir = Path("configs/2-hpo-ensemble")
logging_section = {
    "enabled": True,
    "log_subdir": "logs",
    "verbose": 0,
    "compress": True,
    "upload_to_mlflow": True,
    "max_log_size_mb": 100,
}

for config_file in sorted(configs_dir.glob("ensemble_chemprop_hpo_*.yaml")):
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)

    # Add logging section if not present
    if "logging" not in data:
        data["logging"] = logging_section

        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Updated {config_file.name}")
```

**Location**: After `ray:` section

**Success**: All 100+ files updated, no duplicates

---

### Task 6.4: Update Production Configs (`3-hpo-production/`)

Add logging section to all `ensemble_chemprop_hpo_*.yaml` files in production.

**Approach**: Same script as Task 6.3, pointed at `3-hpo-production/`

**Location**: After `ray:` section

**Success**: All ~100 files updated

---

### Task 6.5: Update Classical Model Configs (`4-more-models/`)

Add logging section to:
- `4-more-models/xgboost.yaml`
- `4-more-models/lightgbm.yaml`
- `4-more-models/catboost.yaml`
- Any other model configs

**Approach**: Manual edits for small count

**Location**: After `mlflow:` section

**Success**: All classical model configs have logging

---

### Task 6.6: Update Curriculum Learning Configs (`curriculum/`)

Add logging section to all curriculum configs:
- `curriculum/chemprop_curriculum.yaml`
- Any other curriculum experiment configs

**Location**: After `ray:` section (if present)

**Success**: Curriculum configs load with logging enabled

---

### Task 6.7: Update Task Affinity Configs (`task-affinity/`)

Add logging section to all task affinity configs:
- `task-affinity/chemprop_task_affinity.yaml`
- Any other task affinity experiment configs

**Location**: After `ray:` section

**Success**: Task affinity configs load with logging enabled

---

### Implementation Approach

Given the large number of files, recommend:

1. **For automated groups** (2-hpo-ensemble, 3-hpo-production): Use Python script
2. **For manual groups** (others): Edit files directly or use find+replace with validation

Example script to run:

```bash
# Validate all YAML files parse correctly after updates
python -c "
import yaml
from pathlib import Path

for yaml_file in Path('configs').rglob('*.yaml'):
    try:
        with open(yaml_file) as f:
            yaml.safe_load(f)
        print(f'✓ {yaml_file}')
    except Exception as e:
        print(f'✗ {yaml_file}: {e}')
"
```

---


## Phase 7: Documentation

### Task 7.1: Create Logging Documentation

Create `docs/guide/logging.rst` with comprehensive logging documentation.

- **Files**:
  - `docs/guide/logging.rst` - NEW: Logging guide

- **Content Outline**:

```rst
Logging and Output Management
=============================

This guide covers logging configuration for HPO and ensemble training.

Overview
--------

By default, Ray Tune produces verbose output that can flood terminals during
long-running HPO and ensemble training jobs. The logging system provides:

- Quiet terminal output with progress indicators
- Full logs captured to files
- Automatic upload to MLflow as artifacts
- Graceful cleanup on interrupts

Configuration
-------------

Add a ``logging`` section to your config:

.. code-block:: yaml

   logging:
     enabled: true
     log_subdir: logs
     verbose: 0
     compress: true
     upload_to_mlflow: true

Verbosity Levels
----------------

- ``0``: Quiet - only progress indicators (recommended for production)
- ``1``: Status - trial start/complete messages
- ``2``: Brief - periodic metric updates
- ``3``: Detailed - full Ray Tune output

Accessing Logs
--------------

After a run completes, logs are available:

1. **Local files**: ``<output_dir>/logs/``
2. **MLflow artifacts**: Run page → Artifacts → logs/

Example CLI Usage
-----------------

.. code-block:: bash

   # HPO with quiet logging
   admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml

   # Terminal shows:
   # 🔬 HPO: 100 trials | metric: val_loss (min)
   # ==================================================
   # [42/100] Trials completed | best val_loss: 0.1234
   # ...
   # ==================================================
   # ✓ HPO complete. Best val_loss: 0.0987
```

- **Success**:
  - Documentation builds without errors
  - Examples are accurate and helpful

- **Dependencies**: Phases 1-4

---

### Task 7.2: Update CLI Documentation

Update CLI command documentation with logging information.

- **Files**:
  - `docs/guide/cli.rst` - MODIFY: Add logging info

- **Changes**: Add note about logging configuration to HPO and ensemble command sections.

- **Success**:
  - CLI docs mention logging options
  - Links to logging guide

- **Dependencies**: Task 7.1
