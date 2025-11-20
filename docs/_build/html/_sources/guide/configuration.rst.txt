Configuration Guide
===================

This page explains how run settings, hyperparameters, and model options are
declared and overridden.

Primary Config File
-------------------

The repository includes YAML configuration(s) under `configs/` (e.g. `configs/xgb.yaml`).
They capture default hyperparameters for model training.

Example (Illustrative)
----------------------

.. code-block:: yaml

  # Example XGBoost config for per-endpoint training
  models:
    xgboost:
      # Docs: https://xgboost.readthedocs.io/en/stable/parameter.html
      objective: "mae" # {"mae", "rmse"}
      early_stopping_rounds: 50
      model_params:
        device: "cuda"
        n_estimators: 500
        learning_rate: 0.3
        min_split_loss: 0
        max_depth: 6
        subsample: 1
        colsample_bytree: 1
        reg_lambda: 1.0
        reg_alpha: 0.0

  training:
    sample_weights:
      enabled: false
      weights:
        default: 1.0

  ray:
    address: "local" # connect to existing cluster (overridden by --ray-address)
    # num_cpus: 8    # limit local Ray runtime to N CPUs (defaults to all)

  data:
    endpoints:
      - "LogD"
      - "KSOL"
      - "HLM CLint"
      - "MLM CLint"
      - "Caco-2 Permeability Papp A>B"
      - "Caco-2 Permeability Efflux"
      - "MPPB"
      - "MBPB"
      - "MGMB"


Overriding Parameters
---------------------

You can override parameters via:

1. CLI flags (if implemented): `--learning-rate 0.1` etc.
2. Environment variables (future extension), e.g. `ADMET_LEARNING_RATE=0.1`.
3. A custom YAML passed with a CLI argument (e.g. `--config configs/custom.yaml`).

Programmatic Loading
--------------------

.. code-block:: python

   import yaml, pathlib

   config_path = pathlib.Path("configs/xgb.yaml")
   config = yaml.safe_load(config_path.read_text())
   params = config["model"]["params"]

   # Pass into trainer or model wrapper

Best Practices
--------------

- Keep model‑specific parameters grouped.
- Include a seed for reproducibility.
- Log the final resolved configuration in training output/artifacts.
- Avoid hard‑coding hyperparameters in Python modules.
