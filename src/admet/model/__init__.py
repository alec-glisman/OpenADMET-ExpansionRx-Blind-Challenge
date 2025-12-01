from __future__ import annotations

import logging
import warnings

__all__ = ["classical", "chemprop"]

# Suppress overly verbose logging from dependencies
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*srun.*")
warnings.filterwarnings("ignore", message=".*num_workers.*")
warnings.filterwarnings("ignore", message=".*Please use `name`.*")
warnings.filterwarnings("ignore", message=".*Please set `input_example`.*")
