from typing import Any, Dict

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def build_model(model_type: str, params: Dict[str, Any]):
    """Build a multi-output regressor wrapping XGB or LGBM."""
    if model_type == "xgboost":
        base = XGBRegressor(**params)
    elif model_type == "lightgbm":
        base = LGBMRegressor(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = MultiOutputRegressor(base)
    return model


def fit_model(model, X, y, sample_weight=None):
    """Fit multi-output model with optional sample weights."""
    model.fit(X, y, sample_weight=sample_weight)
    return model


def predict_model(model, X) -> np.ndarray:
    """Predict with multi-output model."""
    return model.predict(X)
