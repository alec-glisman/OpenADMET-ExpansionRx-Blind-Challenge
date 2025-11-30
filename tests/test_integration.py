import pandas as pd
from omegaconf import OmegaConf

from admet.model.classical.data import augment_quality, get_xyw
from admet.model.classical.models import build_model, fit_model, predict_model
from admet.model.classical.metrics import build_metric_dict


def test_integration_classical_small():
    # Tiny synthetic dataset
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [0.5, 0.6, 0.7, 0.8],
            "quality": ["high", "medium", "low", "high"],
            "y1": [0.1, 0.2, 0.3, 0.4],
            "y2": [1.1, 1.2, 1.3, 1.4],
        }
    )

    df_aug = augment_quality(df, quality_col="quality", quality_weights={"high": 1.0, "medium": 0.5, "low": 0.2})
    X, y, w = get_xyw(df_aug, ["y1", "y2"])

    model = build_model(
        "xgboost",
        {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "booster": "gbtree",
            "objective": "reg:squarederror",
        },
    )
    model = fit_model(model, X, y, sample_weight=w)
    y_pred = predict_model(model, X)
    metrics = build_metric_dict(y, y_pred, [0.5, 0.5], prefix="val")
    assert "val_rmse_weighted" in metrics
