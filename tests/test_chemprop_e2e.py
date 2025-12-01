"""
End-to-end test for ChempropModel training loop; skipped by default.
Set environment variable RUN_E2E=1 to run this more-robust integration test.
"""

import os

import pytest

from admet.model.chemprop.model import ChempropHyperparams, ChempropModel


@pytest.mark.skipif(os.environ.get("RUN_E2E") != "1", reason="E2E tests are skipped by default")
def test_chemprop_model_train_and_predict(sample_dataframe, tmp_path):
    # Use the sample dataframe for training and validation
    train_df = sample_dataframe.iloc[:-2].copy()
    val_df = sample_dataframe.iloc[-2:].copy()

    # Minimal hyperparams to keep training short
    hyperparams = ChempropHyperparams(max_epochs=1, batch_size=4, num_workers=0)

    model = ChempropModel(
        df_train=train_df,
        df_validation=val_df,
        smiles_col="SMILES",
        target_cols=["LogD"],
        target_weights=[1.0],
        progress_bar=False,
        hyperparams=hyperparams,
        mlflow_tracking=False,
    )

    # Train for 1 epoch
    model.fit()

    # Generate predictions
    preds = model.predict(val_df)
    assert not preds.empty
