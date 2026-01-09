"""
Ensemble training for Chemeleon models.

This module re-exports the generic ModelEnsemble from chemprop.ensemble,
which supports all model types including Chemeleon via the ModelRegistry.

Examples
--------
>>> from admet.model.chemeleon.ensemble import ModelEnsemble, main
>>> # Or run directly:
>>> # python -m admet.model.chemeleon.ensemble --config configs/chemeleon_ensemble.yaml
"""

from admet.model.chemprop.ensemble import ModelEnsemble, main, train_ensemble_from_config

__all__ = ["ModelEnsemble", "main", "train_ensemble_from_config"]

if __name__ == "__main__":
    main()
