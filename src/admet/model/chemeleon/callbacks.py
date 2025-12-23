"""Callbacks for Chemeleon model training.

This module provides PyTorch Lightning callbacks for Chemeleon model training,
including gradual unfreezing of encoder and decoder components.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lightning import pytorch as pl

if TYPE_CHECKING:
    from admet.model.config import UnfreezeScheduleConfig

logger = logging.getLogger(__name__)


class GradualUnfreezeCallback(pl.Callback):
    """Callback for gradual unfreezing of encoder and decoder.

    This callback implements scheduled unfreezing of the Chemeleon encoder
    (message passing) and decoder (FFN/predictor) components during training.
    This is useful for transfer learning where we want to initially freeze
    pre-trained components and gradually fine-tune them.

    Parameters
    ----------
    config : UnfreezeScheduleConfig
        Configuration specifying freeze/unfreeze behavior.

    Attributes
    ----------
    config : UnfreezeScheduleConfig
        The unfreeze schedule configuration.
    _encoder_unfrozen : bool
        Whether the encoder has been unfrozen.
    _decoder_unfrozen : bool
        Whether the decoder has been unfrozen.

    Examples
    --------
    >>> from admet.model.config import UnfreezeScheduleConfig
    >>> config = UnfreezeScheduleConfig(
    ...     freeze_encoder=True,
    ...     unfreeze_encoder_epoch=10,
    ... )
    >>> callback = GradualUnfreezeCallback(config)
    >>> trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(self, config: UnfreezeScheduleConfig) -> None:
        """Initialize callback with configuration.

        Parameters
        ----------
        config : UnfreezeScheduleConfig
            Configuration specifying freeze/unfreeze behavior.
        """
        super().__init__()
        self.config = config
        self._encoder_unfrozen = not config.freeze_encoder
        self._decoder_unfrozen = not config.freeze_decoder_initially

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check and apply unfreezing at epoch start.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The Lightning module being trained.
        """
        epoch = trainer.current_epoch

        # Unfreeze encoder at specified epoch
        if not self._encoder_unfrozen and self.config.unfreeze_encoder_epoch is not None:
            if epoch >= self.config.unfreeze_encoder_epoch:
                self._unfreeze_encoder(pl_module)
                self._encoder_unfrozen = True
                logger.info(f"Unfroze encoder at epoch {epoch}")

        # Unfreeze decoder at specified epoch
        if not self._decoder_unfrozen and self.config.unfreeze_decoder_epoch is not None:
            if epoch >= self.config.unfreeze_decoder_epoch:
                self._unfreeze_decoder(pl_module)
                self._decoder_unfrozen = True
                logger.info(f"Unfroze decoder at epoch {epoch}")

    def _unfreeze_encoder(self, pl_module: pl.LightningModule) -> None:
        """Unfreeze message passing encoder.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The Lightning module containing the encoder.
        """
        # Try different attribute names for the encoder
        encoder_attrs = ["message_passing", "encoder", "mp"]
        for attr in encoder_attrs:
            if hasattr(pl_module, attr):
                encoder = getattr(pl_module, attr)
                for param in encoder.parameters():
                    param.requires_grad = True
                encoder.train()
                return

        logger.warning("Could not find encoder to unfreeze")

    def _unfreeze_decoder(self, pl_module: pl.LightningModule) -> None:
        """Unfreeze predictor/FFN decoder.

        Parameters
        ----------
        pl_module : pl.LightningModule
            The Lightning module containing the decoder.
        """
        # Try different attribute names for the decoder
        decoder_attrs = ["predictor", "ffn", "decoder", "head"]
        for attr in decoder_attrs:
            if hasattr(pl_module, attr):
                decoder = getattr(pl_module, attr)
                for param in decoder.parameters():
                    param.requires_grad = True
                decoder.train()
                return

        logger.warning("Could not find decoder to unfreeze")

    @property
    def is_encoder_frozen(self) -> bool:
        """Check if encoder is currently frozen."""
        return not self._encoder_unfrozen

    @property
    def is_decoder_frozen(self) -> bool:
        """Check if decoder is currently frozen."""
        return not self._decoder_unfrozen
