"""admet.train.xgb_train
=========================

XGBoost‑specific trainer implementations for multi‑endpoint ADMET regression.

Components
----------
``XGBoostTrainer``
    Concrete :class:`admet.train.base_trainer.BaseModelTrainer` implementing
    fingerprint feature extraction, endpoint target handling with masks, and
    metric computation via :func:`admet.evaluate.metrics.compute_metrics_log_and_linear`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from admet.model.base import ModelProtocol
from admet.train.base import BaseModelTrainer, FeaturizationMethod

logger = logging.getLogger(__name__)


class XGBoostTrainer(BaseModelTrainer):
    """Single-dataset trainer implementation using XGBoost.

    This class implements :class:`~admet.train.base_trainer.BaseModelTrainer`
    for XGBoost backend models, delegates per-endpoint training to
    :class:`admet.model.xgb_wrapper.XGBoostMultiEndpoint`, and preserves
    existing metrics and artifact formats used by the higher-level
    orchestration logic.

    Notes
    -----
    The class uses dataset fingerprints as features extracted from
    :class:`admet.data.load.LoadedDataset` and predicts all endpoints using
    independent XGBoost regressors per endpoint.
    """

    def __init__(
        self,
        model_cls: Type[ModelProtocol],
        *,
        model_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        mixed_precision: bool = False,
        fingerprint_config=None,
    ) -> None:
        super().__init__(
            model_cls,
            model_params=model_params,
            seed=seed,
            device=device,
            mixed_precision=mixed_precision,
            fingerprint_config=fingerprint_config,
        )

        self.featurization = FeaturizationMethod.MORGAN_FP


__all__ = [
    "XGBoostTrainer",
]
