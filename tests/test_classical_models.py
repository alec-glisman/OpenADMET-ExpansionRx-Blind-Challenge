import pandas as pd
from admet.model.classical.data import augment_quality, get_xyw, make_folds


def test_augment_quality_and_get_xyw():
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
    assert "quality_bucket" in df_aug.columns
    assert "sample_weight" in df_aug.columns

    X, y, w = get_xyw(df_aug, ["y1", "y2"])
    assert X.shape == (4, 2)
    assert y.shape == (4, 2)
    assert w.shape == (4,)


def test_make_folds():
    df = pd.DataFrame({"x": range(10)})
    folds = make_folds(df, n_splits=5, random_state=0, shuffle=True)
    assert len(folds) == 5
    all_indices = []
    for tr, val in folds:
        all_indices.extend(val.tolist())
    assert sorted(all_indices) == list(range(10))
