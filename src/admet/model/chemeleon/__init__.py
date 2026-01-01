"""Chemeleon foundation model for molecular property prediction.

This module provides a wrapper around the Chemeleon pre-trained molecular encoder
for ADMET property prediction with optional fine-tuning capabilities.
"""

from admet.model.chemeleon.callbacks import GradualUnfreezeCallback
from admet.model.chemeleon.hpo import ChemeleonHPO
from admet.model.chemeleon.model import ChemeleonModel

__all__ = [
    "ChemeleonHPO",
    "ChemeleonModel",
    "GradualUnfreezeCallback",
]
