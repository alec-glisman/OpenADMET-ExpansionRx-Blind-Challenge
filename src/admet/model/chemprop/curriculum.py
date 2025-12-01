import math
from typing import Any, Dict, List, Optional

# torch not required directly here; trainer metrics may include torch tensors
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping


class CurriculumState:
    """Simple quality-aware curriculum state.

    Phases:
      - warmup: focus on high-quality data
      - expand: include more medium-quality
      - robust: include some low-quality
      - polish: re-focus on high-quality
    """

    def __init__(self, qualities: Optional[List[str]] = None, patience: int = 3):
        """Create a curriculum that can support arbitrary quality labels.

        Parameters
        ----------
        qualities
            Ordered list of quality levels, highest-to-lowest (e.g. ["high","medium","low"]).
        patience
            Number of epochs with no improvement before moving to the next phase.
        """
        if qualities is None:
            qualities = ["high", "medium", "low"]
        if not qualities:
            raise ValueError("`qualities` must be a non-empty list of names")

        self.qualities = list(qualities)
        self.phase = "warmup"
        # default initial weights derived from provided qualities
        # Start in warmup phase weights
        self.weights = self._weights_for_phase("warmup")

        self.best_val_top = float("inf")
        self.best_epoch = 0
        self.patience = patience

    def target_metric_key(self) -> str:
        """Return the metric key monitored by the curriculum (val_loss).

        Uses overall validation loss (not per-quality loss) for phase transitions
        to ensure the model improves across all quality levels.
        """
        return "val_loss"

    def update_from_val_top(self, epoch: int, top_loss: float):
        if top_loss < self.best_val_top - 1e-4:
            self.best_val_top = top_loss
            self.best_epoch = epoch

    def maybe_advance_phase(self, epoch: int):
        """Advance to the next phase when patience has passed with no top-quality improvement.

        The progression adapts to however many `qualities` are provided. For 1-quality
        datasets this will simply do warmup -> polish (return focus to top quality).
        For 2-quality datasets: warmup -> expand -> polish.
        For >=3: warmup -> expand -> robust -> polish.
        """
        if epoch - self.best_epoch < self.patience:
            return

        n = len(self.qualities)
        phases = ["warmup"]
        if n >= 2:
            phases.append("expand")
        if n >= 3:
            phases.append("robust")
        phases.append("polish")

        # determine current phase index
        try:
            idx = phases.index(self.phase)
        except ValueError:
            idx = 0

        # move to the next phase unless we are already at 'polish'
        if idx < len(phases) - 1:
            idx += 1
            self.phase = phases[idx]
        # compute weights based on phase and number of qualities
        self.weights = self._weights_for_phase(self.phase)

    def _weights_for_phase(self, phase: str) -> dict:
        """Return a weight mapping for the given phase adapted to the number of qualities."""
        n = len(self.qualities)

        def _top_k_weights(k: int, top_weight: float) -> dict:
            # k is number of qualities to include starting from top
            weights = {q: 0.0 for q in self.qualities}
            if k <= 0:
                return weights
            if k == 1:
                weights[self.qualities[0]] = 1.0
                return weights
            top = float(top_weight)
            remaining = 1.0 - top
            per_other = remaining / (k - 1)
            for idx_q in range(k):
                q = self.qualities[idx_q]
                weights[q] = top if idx_q == 0 else per_other
            return weights

        # Exact schedules for n == 1/2/3 to match original algorithm
        if n == 1:
            return {self.qualities[0]: 1.0}
        if n == 2:
            if phase == "warmup":
                return {self.qualities[0]: 0.9, self.qualities[1]: 0.1}
            if phase == "expand":
                return {self.qualities[0]: 0.6, self.qualities[1]: 0.4}
            if phase == "polish":
                return {self.qualities[0]: 1.0, self.qualities[1]: 0.0}
            # fallback
            return {self.qualities[0]: 1.0, self.qualities[1]: 0.0}
        if n == 3:
            if phase == "warmup":
                return {self.qualities[0]: 0.9, self.qualities[1]: 0.1, self.qualities[2]: 0.0}
            if phase == "expand":
                return {self.qualities[0]: 0.6, self.qualities[1]: 0.35, self.qualities[2]: 0.05}
            if phase == "robust":
                return {self.qualities[0]: 0.4, self.qualities[1]: 0.4, self.qualities[2]: 0.2}
            if phase == "polish":
                return {self.qualities[0]: 1.0, self.qualities[1]: 0.0, self.qualities[2]: 0.0}
        # fallback for n > 3: equal weighting across qualities
        equal = 1.0 / n
        return {q: equal for q in self.qualities}

    def sampling_probs(self):
        s = sum(self.weights.values())
        return {k: v / s for k, v in self.weights.items()}


class CurriculumCallback(pl.Callback):
    """Update curriculum state based on validation loss metric.

    The metric key monitored defaults to 'val_loss' (overall validation loss)
    but can be overridden via the monitor_metric parameter.

    Phase transitions are logged with epoch number and step for tracking
    curriculum progression during training.

    Parameters
    ----------
    curr_state : CurriculumState
        The curriculum state object to update.
    monitor_metric : str, optional
        Metric key to monitor for phase transitions. Defaults to 'val_loss'.
    reset_early_stopping_on_phase_change : bool, default=False
        Whether to reset early stopping patience when advancing phases.
    log_per_quality_metrics : bool, default=True
        Whether to log per-quality validation metrics.
    quality_labels : List[str], optional
        Quality labels for validation samples. Required for per-quality metrics.
    """

    def __init__(
        self,
        curr_state: CurriculumState,
        monitor_metric: Optional[str] = None,
        reset_early_stopping_on_phase_change: bool = False,
        log_per_quality_metrics: bool = True,
        quality_labels: Optional[List[str]] = None,
    ):
        super().__init__()
        self.curr_state = curr_state
        self.monitor_metric = monitor_metric
        self._previous_phase = curr_state.phase
        self.reset_early_stopping_on_phase_change = reset_early_stopping_on_phase_change
        self.log_per_quality_metrics = log_per_quality_metrics
        self.quality_labels = quality_labels

        # Cache quality indices for efficient per-quality metric computation
        self._quality_indices: Optional[Dict[str, List[int]]] = None
        if quality_labels is not None:
            self._quality_indices = {}
            for i, label in enumerate(quality_labels):
                if label not in self._quality_indices:
                    self._quality_indices[label] = []
                self._quality_indices[label].append(i)

    def _reset_early_stopping(self, trainer: pl.Trainer) -> None:
        """Reset early stopping callback's wait counter and best score."""
        callbacks = getattr(trainer, "callbacks", [])
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                callback.wait_count = 0
                # Optionally reset best score to allow model to re-establish baseline
                # callback.best_score = callback.mode_dict[callback.mode](torch.tensor(float('inf')))
                import logging

                logger = logging.getLogger("admet.model.chemprop.curriculum")
                logger.info("Reset early stopping patience after curriculum phase change")
                break

    def _log_per_quality_metrics(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,
        metrics: Dict[str, Any],
    ) -> None:
        """Log per-quality validation metrics if available."""
        if not self.log_per_quality_metrics:
            return

        # Look for per-quality metrics that may have been computed by the model
        for quality in self.curr_state.qualities:
            # Check for metrics like val_mae_high, val_rmse_medium, etc.
            for base_metric in ["val_mae", "val_rmse", "val_loss"]:
                quality_metric_key = f"{base_metric}_{quality}"
                if quality_metric_key in metrics:
                    val = metrics[quality_metric_key]
                    try:
                        v = val.item() if hasattr(val, "item") else float(val)
                        pl_module.log(
                            quality_metric_key,
                            v,
                            on_step=False,
                            on_epoch=True,
                        )
                    except Exception:
                        pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handle validation epoch end: update state, check phase, log metrics."""
        metrics = trainer.callback_metrics
        metric_key = self.monitor_metric or self.curr_state.target_metric_key()
        val_top = metrics.get(metric_key)
        if val_top is None:
            return
        # tolerate both torch.Tensor and float/np scalar
        try:
            v = val_top.item() if hasattr(val_top, "item") else float(val_top)
        except Exception:
            return
        if math.isnan(v):
            return

        epoch = trainer.current_epoch
        global_step = trainer.global_step
        self.curr_state.update_from_val_top(epoch, float(v))
        self.curr_state.maybe_advance_phase(epoch)

        # Log per-quality metrics
        self._log_per_quality_metrics(trainer, pl_module, metrics)

        # Log phase transitions
        if self.curr_state.phase != self._previous_phase:
            import logging

            logger = logging.getLogger("admet.model.chemprop.curriculum")
            logger.info(
                "Curriculum phase transition: %s -> %s at epoch %d (step %d), " "val_loss=%.4f, weights=%s",
                self._previous_phase,
                self.curr_state.phase,
                epoch,
                global_step,
                v,
                self.curr_state.weights,
            )

            # Log phase transition to trainer's logger (goes to MLflow if enabled)
            phase_idx = {"warmup": 0, "expand": 1, "robust": 2, "polish": 3}.get(self.curr_state.phase, -1)
            pl_module.log("curriculum_phase", float(phase_idx), on_step=False, on_epoch=True)
            pl_module.log("curriculum_phase_epoch", float(epoch), on_step=False, on_epoch=True)

            # Log current weights for each quality
            for quality, weight in self.curr_state.weights.items():
                pl_module.log(
                    f"curriculum_weight_{quality}",
                    float(weight),
                    on_step=False,
                    on_epoch=True,
                )

            # Reset early stopping if configured
            if self.reset_early_stopping_on_phase_change:
                self._reset_early_stopping(trainer)

            self._previous_phase = self.curr_state.phase
