Modeling Guide
==============

This page describes model wrappers and how they integrate with training and
configuration components.

Wrapper Abstraction
-------------------

Wrappers in `admet.model` standardize interactions with underlying libraries
(e.g. XGBoost, LightGBM) and expose a consistent interface for:

- Initialization with hyperparameters
- Fitting on training data
- Predicting probabilities or continuous values
- Persisting / loading artifacts

Example Flow
------------

.. code-block:: python

   from admet.model.xgb_wrapper import XGBoostWrapper

   model = XGBoostWrapper(params={"max_depth": 6, "learning_rate": 0.05})
   model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
   preds = model.predict_proba(X_test)

Trainer Interaction
-------------------

Training modules in `admet.train` coordinate:

1. Dataset assembly and splitting
2. Model wrapper instantiation
3. Metric evaluation callbacks
4. Artifact persistence (e.g. best iteration parameters)

Artifact Layout
---------------

Trained models and related metadata may be stored under directories like:

- `temp/xgb_artifacts/`
- `temp/xgb_artifacts/ensemble/high_quality/`

Common Wrapper Features (Conceptual)
------------------------------------

- Parameter Validation: Ensures provided hyperparameters are acceptable.
- Fit/Predict API: Align naming across implementations.
- Probability Output: For classification tasks, `predict_proba` returns class probabilities.
- Early Stopping Support: Pass validation sets to underlying library for early stopping.
- Serialization: Save booster/model object plus config JSON/YAML.

Extending with a New Model
--------------------------

To add a model type:

1. Create a new file `my_model_wrapper.py` in `admet/model`.
2. Implement a class exposing `fit`, `predict`, and optional `predict_proba`.
3. Handle parameter ingestion (store `self.params`).
4. Add serialization helpers (`save`, `load`).
5. Update CLI/train orchestration to recognize new `--model-type`.

Evaluation Hooks
----------------

After training, metrics from `admet.evaluate.metrics` are applied to predictions
on validation/test sets. Store results alongside artifacts for reproducibility.

XGBoost Hyperparameter Considerations
--------------------------------------

- Depth / Complexity: Increase cautiously to avoid overfitting.
- Learning Rate: Lower rates often improve generalization (paired with higher rounds).
- Regularization: Leverage `reg_lambda`, `reg_alpha` to stabilize models.
- Sampling: `subsample` and `colsample_bytree` for diversity.


Cross-References
----------------

- See :doc:`configuration` for passing hyperparameters.
