import math
from typing import Any, Dict, List, Optional

import numpy as np

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

    def on_train_start(self, trainer: Any, pl_module: pl.LightningModule) -> None:
        """Log initial curriculum state at the start of training."""
        import logging

        logger = logging.getLogger("admet.model.chemprop.curriculum")
        logger.info(
            "Starting curriculum learning: phase=%s, weights=%s",
            self.curr_state.phase,
            self.curr_state.weights,
        )

        # Log initial curriculum state to MLflow
        phase_idx = {"warmup": 0, "expand": 1, "robust": 2, "polish": 3}.get(self.curr_state.phase, -1)
        pl_module.log("curriculum/phase", float(phase_idx), on_step=False, on_epoch=True)

        for quality, weight in self.curr_state.weights.items():
            pl_module.log(
                f"curriculum/weight/{quality}",
                float(weight),
                on_step=False,
                on_epoch=True,
            )

    def _reset_early_stopping(self, trainer: Any) -> None:
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
        trainer: Any,  # noqa: ARG002
        pl_module: pl.LightningModule,
        metrics: Dict[str, Any],
    ) -> None:
        """Log per-quality validation metrics if available.

        Metrics are logged with hierarchical naming: val/<metric>/<quality>
        For example: val/mae/high, val/rmse/medium, val/loss/low
        """
        if not self.log_per_quality_metrics:
            return

        # Look for per-quality metrics that may have been computed by the model
        for quality in self.curr_state.qualities:
            # Check for metrics like val/mae/high, val/rmse/medium, etc.
            # Also check legacy format for backward compatibility
            for base_metric in ["mae", "rmse", "loss"]:
                # New hierarchical format: val/<metric>/<quality>
                new_key = f"val/{base_metric}/{quality}"
                # Legacy underscore format
                legacy_key = f"val_{base_metric}_{quality}"

                metric_key = new_key if new_key in metrics else legacy_key
                if metric_key in metrics:
                    val = metrics[metric_key]
                    try:
                        v = val.item() if hasattr(val, "item") else float(val)
                        pl_module.log(
                            new_key,
                            v,
                            on_step=False,
                            on_epoch=True,
                        )
                    except Exception:
                        pass

    def on_validation_epoch_end(self, trainer: Any, pl_module: pl.LightningModule) -> None:
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

        # Always log current curriculum state to MLflow (not just on transitions)
        phase_idx = {"warmup": 0, "expand": 1, "robust": 2, "polish": 3}.get(self.curr_state.phase, -1)
        pl_module.log("curriculum/phase", float(phase_idx), on_step=False, on_epoch=True)

        # Log current weights for each quality
        for quality, weight in self.curr_state.weights.items():
            pl_module.log(
                f"curriculum/weight/{quality}",
                float(weight),
                on_step=False,
                on_epoch=True,
            )

        # Log phase transitions (with additional details)
        if self.curr_state.phase != self._previous_phase:
            import logging

            import mlflow

            logger = logging.getLogger("admet.model.chemprop.curriculum")
            logger.info(
                "Curriculum phase transition: %s -> %s at epoch %d (step %d), val_loss=%.4f, weights=%s",
                self._previous_phase,
                self.curr_state.phase,
                epoch,
                global_step,
                v,
                self.curr_state.weights,
            )

            # Log phase transition metadata
            pl_module.log("curriculum/phase_epoch", float(epoch), on_step=False, on_epoch=True)
            pl_module.log("curriculum/val_loss_at_transition", float(v), on_step=False, on_epoch=True)

            # Log MLflow tag for curriculum stage transition (clear marker for training curves)
            try:
                if mlflow.active_run():
                    mlflow.set_tag(
                        f"curriculum_transition_epoch_{epoch}",
                        f"{self._previous_phase}_to_{self.curr_state.phase}",
                    )
                    # Also log as a metric for visibility in training curves
                    mlflow.log_metric("curriculum/transition", float(phase_idx), step=global_step)
            except Exception:
                pass  # Silently ignore MLflow tagging failures

            # Reset early stopping if configured
            if self.reset_early_stopping_on_phase_change:
                self._reset_early_stopping(trainer)

            self._previous_phase = self.curr_state.phase


class PerQualityMetricsCallback(pl.Callback):
    """Compute and log per-quality metrics during training and validation.

    This callback computes predictions at the end of each validation epoch
    and calculates MAE, MSE, RMSE for each data quality level.

    Since Chemprop's validation_step doesn't return outputs, this callback
    runs the model's predict_step on the validation dataloader at the end
    of each epoch to compute per-quality metrics.

    Metrics are logged with hierarchical naming for clear organization in MLflow:
    - val/<metric>/<quality> (e.g., val/mae/high, val/rmse/medium)
    - val/count/<quality> (number of samples per quality)

    Parameters
    ----------
    val_quality_labels : List[str]
        Quality label for each validation sample (e.g., ["high", "medium", "low", ...]).
    qualities : List[str]
        Ordered list of quality levels to track (e.g., ["high", "medium", "low"]).
    target_cols : List[str], optional
        Target column names for per-target metrics. If provided, also logs
        val/<metric>/<quality>/<target>, e.g., val/mae/high/LogD.
    compute_every_n_epochs : int, default=1
        Compute per-quality metrics every N epochs (to reduce overhead).

    Examples
    --------
    >>> callback = PerQualityMetricsCallback(
    ...     val_quality_labels=df_val["Quality"].tolist(),
    ...     qualities=["high", "medium", "low"],
    ...     target_cols=["LogD", "KSOL"],
    ... )
    >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        val_quality_labels: List[str],
        qualities: List[str],
        target_cols: Optional[List[str]] = None,
        compute_every_n_epochs: int = 1,
        # Legacy parameter names for backward compatibility
        quality_labels: Optional[List[str]] = None,
        train_quality_labels: Optional[List[str]] = None,  # noqa: ARG002
    ):
        super().__init__()
        # Support legacy parameter name
        if quality_labels is not None and val_quality_labels is None:
            val_quality_labels = quality_labels
        self.val_quality_labels = val_quality_labels
        self.qualities = qualities
        self.target_cols = target_cols
        self.compute_every_n_epochs = compute_every_n_epochs

        # Build quality indices for efficient grouping (validation)
        self._val_quality_indices: Dict[str, List[int]] = {q: [] for q in qualities}
        for i, label in enumerate(val_quality_labels):
            if label in self._val_quality_indices:
                self._val_quality_indices[label].append(i)

    def _compute_and_log_metrics(
        self,
        pl_module: pl.LightningModule,
        all_preds: np.ndarray,
        all_targets: np.ndarray,
        quality_indices: Dict[str, List[int]],
        split: str,
    ) -> None:
        """Compute and log per-quality metrics for a given split (train/val).

        Metrics are logged with format: <split>/<metric>/<quality>
        e.g., val/mae/high, train/rmse/medium
        """
        import mlflow

        for quality in self.qualities:
            indices = quality_indices.get(quality, [])
            if not indices:
                continue

            q_preds = all_preds[indices]
            q_targets = all_targets[indices]

            # Handle NaN values
            valid_mask = ~(np.isnan(q_preds) | np.isnan(q_targets))
            if valid_mask.ndim > 1:
                valid_mask = valid_mask.any(axis=1)

            if not valid_mask.any():
                continue

            q_preds_valid = q_preds[valid_mask]
            q_targets_valid = q_targets[valid_mask]

            # Flatten for overall metrics
            if q_preds_valid.ndim > 1:
                flat_mask = ~(np.isnan(q_preds_valid) | np.isnan(q_targets_valid))
                q_preds_flat = q_preds_valid[flat_mask]
                q_targets_flat = q_targets_valid[flat_mask]
            else:
                q_preds_flat = q_preds_valid.flatten()
                q_targets_flat = q_targets_valid.flatten()

            if len(q_preds_flat) == 0:
                continue

            # Compute metrics
            mae = float(np.mean(np.abs(q_preds_flat - q_targets_flat)))
            mse = float(np.mean((q_preds_flat - q_targets_flat) ** 2))
            rmse = float(np.sqrt(mse))

            # Log with hierarchical naming: <split>/<metric>/<quality>
            # Use pl_module.log for Lightning integration
            pl_module.log(f"{split}/mae/{quality}", mae, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/mse/{quality}", mse, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/rmse/{quality}", rmse, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/count/{quality}", float(len(q_preds_flat)), on_step=False, on_epoch=True)

            # Also log directly to MLflow for immediate visibility
            try:
                if mlflow.active_run():
                    step = pl_module.current_epoch if hasattr(pl_module, "current_epoch") else 0
                    mlflow.log_metric(f"{split}/mae/{quality}", mae, step=step)
                    mlflow.log_metric(f"{split}/mse/{quality}", mse, step=step)
                    mlflow.log_metric(f"{split}/rmse/{quality}", rmse, step=step)
                    mlflow.log_metric(f"{split}/count/{quality}", float(len(q_preds_flat)), step=step)
            except Exception:
                pass

            # Per-target metrics: <split>/<metric>/<quality>/<target>
            if self.target_cols is not None and q_preds_valid.ndim > 1:
                for t_idx, target in enumerate(self.target_cols):
                    if t_idx >= q_preds_valid.shape[1]:
                        continue

                    t_preds = q_preds_valid[:, t_idx]
                    t_targets = q_targets_valid[:, t_idx]

                    t_valid = ~(np.isnan(t_preds) | np.isnan(t_targets))
                    if not t_valid.any():
                        continue

                    t_preds_v = t_preds[t_valid]
                    t_targets_v = t_targets[t_valid]

                    t_mae = float(np.mean(np.abs(t_preds_v - t_targets_v)))
                    t_mse = float(np.mean((t_preds_v - t_targets_v) ** 2))
                    t_rmse = float(np.sqrt(t_mse))

                    pl_module.log(f"{split}/mae/{quality}/{target}", t_mae, on_step=False, on_epoch=True)
                    pl_module.log(f"{split}/mse/{quality}/{target}", t_mse, on_step=False, on_epoch=True)
                    pl_module.log(f"{split}/rmse/{quality}/{target}", t_rmse, on_step=False, on_epoch=True)

                    try:
                        if mlflow.active_run():
                            step = pl_module.current_epoch if hasattr(pl_module, "current_epoch") else 0
                            mlflow.log_metric(f"{split}/mae/{quality}/{target}", t_mae, step=step)
                            mlflow.log_metric(f"{split}/mse/{quality}/{target}", t_mse, step=step)
                            mlflow.log_metric(f"{split}/rmse/{quality}/{target}", t_rmse, step=step)
                    except Exception:
                        pass

    def on_validation_epoch_end(self, trainer: Any, pl_module: pl.LightningModule) -> None:
        """Compute and log per-quality validation metrics at the end of validation epoch."""
        import logging

        import torch

        logger = logging.getLogger("admet.model.chemprop.curriculum")
        logger.debug("PerQualityMetricsCallback.on_validation_epoch_end called")

        # Skip if not the right epoch
        current_epoch = trainer.current_epoch
        logger.debug("Current epoch: %d, compute_every_n_epochs: %d", current_epoch, self.compute_every_n_epochs)
        if current_epoch % self.compute_every_n_epochs != 0:
            logger.debug("Skipping epoch %d (not divisible by %d)", current_epoch, self.compute_every_n_epochs)
            return

        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        logger.debug("Validation dataloader: %s, is_none: %s", type(val_dataloader), val_dataloader is None)
        if val_dataloader is None:
            logger.debug("No validation dataloader available")
            return

        # Handle single dataloader case
        if not isinstance(val_dataloader, list):
            val_dataloader = [val_dataloader]

        logger.debug("Number of dataloaders: %d", len(val_dataloader))
        if not val_dataloader:
            return

        logger.debug("Quality labels count: %d", len(self.val_quality_labels))
        logger.debug("Target columns: %s", self.target_cols)

        try:
            # Collect predictions and targets from validation dataloader
            all_preds = []
            all_targets = []

            pl_module.eval()
            logger.debug("Starting batch iteration for predictions...")
            batch_count = 0
            with torch.no_grad():
                for dl_idx, dl in enumerate(val_dataloader):
                    logger.debug("Processing dataloader %d/%d", dl_idx + 1, len(val_dataloader))
                    for batch_idx, batch in enumerate(dl):
                        batch_count += 1
                        # Chemprop batch format: (bmg, V_d, X_d, targets, weights, lt_mask, gt_mask)
                        if len(batch) >= 4:
                            bmg, V_d, X_d, targets = batch[0], batch[1], batch[2], batch[3]
                            logger.debug("  Batch %d: targets shape=%s", batch_idx, targets.shape)

                            # Move to device
                            # Note: BatchMolGraph.to() modifies in-place and returns None
                            device = next(pl_module.parameters()).device
                            if hasattr(bmg, "to"):
                                bmg.to(device)  # In-place modification
                            if V_d is not None and hasattr(V_d, "to"):
                                V_d = V_d.to(device)
                            if X_d is not None and hasattr(X_d, "to"):
                                X_d = X_d.to(device)

                            # Get predictions
                            preds = pl_module(bmg, V_d, X_d)
                            logger.debug(f"  Predictions shape: {preds.shape}")

                            # Handle multi-target case where preds has extra dimension
                            if preds.ndim == 3 and preds.shape[-1] == 1:
                                preds = preds[..., 0]

                            all_preds.append(preds.cpu().numpy())
                            all_targets.append(targets.cpu().numpy())

            logger.debug(f"Processed {batch_count} batches total")
            logger.debug(f"Collected {len(all_preds)} prediction batches")
            if not all_preds:
                logger.debug("No predictions collected from validation dataloader")
                return

            # Concatenate all batches
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            logger.debug(f"Concatenated predictions shape: {all_preds_np.shape}")
            logger.debug(f"Concatenated targets shape: {all_targets_np.shape}")

            # Verify sample count matches
            if len(all_preds_np) != len(self.val_quality_labels):
                logger.warning(
                    "Mismatch: %d predictions vs %d quality labels - cannot compute per-quality metrics!",
                    len(all_preds_np),
                    len(self.val_quality_labels),
                )
                return

            # Compute and log metrics
            logger.debug("Computing per-quality metrics...")
            self._compute_and_log_metrics(pl_module, all_preds_np, all_targets_np, self._val_quality_indices, "val")
            logger.info("Per-quality validation metrics computed for epoch %d", current_epoch)

        except Exception as e:
            import traceback

            logger.warning("Failed to compute per-quality metrics: %s\n%s", e, traceback.format_exc())


class JointSamplerStatsCallback(pl.Callback):
    """Log JointSampler weight statistics to MLflow.

    This callback logs sampling weight statistics (entropy, effective samples, etc.)
    to MLflow at the start of each training epoch. The statistics are computed by
    the JointSampler during data loading.

    Metrics logged:
    - sampling/weight_min - Minimum sample weight
    - sampling/weight_max - Maximum sample weight
    - sampling/weight_mean - Mean sample weight
    - sampling/entropy - Entropy of weight distribution (higher = more uniform)
    - sampling/effective_samples - Effective number of samples (1/sum(w²))

    Parameters
    ----------
    sampler : JointSampler
        The JointSampler instance from which to extract statistics.
    """

    def __init__(self, sampler: Any) -> None:
        super().__init__()
        self.sampler = sampler

    def on_train_epoch_start(self, trainer: Any, pl_module: pl.LightningModule) -> None:
        """Log sampling statistics at the start of each training epoch."""
        # Check if sampler has computed statistics
        if not hasattr(self.sampler, "_last_weight_stats") or self.sampler._last_weight_stats is None:
            return

        stats = self.sampler._last_weight_stats

        # Log to PyTorch Lightning (which forwards to MLflow)
        for key, value in stats.items():
            pl_module.log(
                f"sampling/weight_{key}",
                float(value),
                on_step=False,
                on_epoch=True,
            )
