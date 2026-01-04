"""Tests for model-agnostic HPO search space."""

from __future__ import annotations

import pytest
from ray import tune

from admet.model.hpo import (
    CatBoostSearchSpace,
    ChemeleonSearchSpace,
    FingerprintSearchSpace,
    HPOSearchSpaceConfig,
    LightGBMSearchSpace,
    ParameterSpace,
    XGBoostSearchSpace,
    get_default_catboost_search_space,
    get_default_chemeleon_search_space,
    get_default_fingerprint_search_space,
    get_default_lightgbm_search_space,
    get_default_xgboost_search_space,
    get_search_space_for_model,
)
from admet.model.hpo.search_space import (
    _build_parameter_space,
    build_catboost_search_space,
    build_chemeleon_search_space,
    build_fingerprint_search_space,
    build_lightgbm_search_space,
    build_search_space,
    build_xgboost_search_space,
)


class TestParameterSpace:
    """Tests for ParameterSpace conversion."""

    def test_uniform(self):
        """Test uniform distribution."""
        param = ParameterSpace(type="uniform", low=0.0, high=1.0)
        result = _build_parameter_space(param)
        assert isinstance(result, tune.search.sample.Float)

    def test_loguniform(self):
        """Test loguniform distribution."""
        param = ParameterSpace(type="loguniform", low=1e-5, high=1e-1)
        result = _build_parameter_space(param)
        assert isinstance(result, tune.search.sample.Float)

    def test_choice(self):
        """Test choice distribution."""
        param = ParameterSpace(type="choice", values=[1, 2, 3])
        result = _build_parameter_space(param)
        assert isinstance(result, tune.search.sample.Categorical)

    def test_randint(self):
        """Test randint distribution."""
        param = ParameterSpace(type="randint", low=1, high=10)
        result = _build_parameter_space(param)
        assert isinstance(result, tune.search.sample.Integer)

    def test_quniform(self):
        """Test quniform distribution."""
        param = ParameterSpace(type="quniform", low=0.0, high=1.0, q=0.1)
        result = _build_parameter_space(param)
        assert isinstance(result, tune.search.sample.Float)

    def test_missing_bounds_raises(self):
        """Test that missing bounds raises error."""
        param = ParameterSpace(type="uniform", low=0.0)  # Missing high
        with pytest.raises(ValueError, match="requires 'low' and 'high'"):
            _build_parameter_space(param)

    def test_unknown_type_raises(self):
        """Test that unknown type raises error."""
        param = ParameterSpace(type="unknown", low=0.0, high=1.0)
        with pytest.raises(ValueError, match="Unknown parameter type"):
            _build_parameter_space(param)


class TestDefaultSearchSpaces:
    """Tests for default search space functions."""

    def test_xgboost_defaults(self):
        """Test XGBoost default search space."""
        space = get_default_xgboost_search_space()
        assert isinstance(space, XGBoostSearchSpace)
        assert space.n_estimators is not None
        assert space.learning_rate is not None

    def test_lightgbm_defaults(self):
        """Test LightGBM default search space."""
        space = get_default_lightgbm_search_space()
        assert isinstance(space, LightGBMSearchSpace)
        assert space.n_estimators is not None
        assert space.num_leaves is not None

    def test_catboost_defaults(self):
        """Test CatBoost default search space."""
        space = get_default_catboost_search_space()
        assert isinstance(space, CatBoostSearchSpace)
        assert space.iterations is not None
        assert space.depth is not None

    def test_chemeleon_defaults(self):
        """Test Chemeleon default search space."""
        space = get_default_chemeleon_search_space()
        assert isinstance(space, ChemeleonSearchSpace)
        assert space.ffn_hidden_dim is not None
        assert space.learning_rate is not None

    def test_fingerprint_defaults(self):
        """Test fingerprint default search space."""
        space = get_default_fingerprint_search_space()
        assert isinstance(space, FingerprintSearchSpace)
        assert space.fp_type is not None

    def test_get_search_space_for_model(self):
        """Test getting search space by model type."""
        for model_type in ["xgboost", "lightgbm", "catboost", "chemeleon"]:
            space = get_search_space_for_model(model_type)
            assert space is not None

    def test_get_search_space_unknown_model_raises(self):
        """Test that unknown model type raises error."""
        with pytest.raises(ValueError, match="not supported"):
            get_search_space_for_model("unknown")


class TestBuildSearchSpace:
    """Tests for building Ray Tune search spaces."""

    def test_build_xgboost_search_space(self):
        """Test building XGBoost search space."""
        config = XGBoostSearchSpace(
            n_estimators=ParameterSpace(type="randint", low=50, high=200),
            learning_rate=ParameterSpace(type="loguniform", low=0.01, high=0.3),
        )
        space = build_xgboost_search_space(config)

        assert "n_estimators" in space
        assert "learning_rate" in space
        assert len(space) == 2

    def test_build_lightgbm_search_space(self):
        """Test building LightGBM search space."""
        config = LightGBMSearchSpace(
            n_estimators=ParameterSpace(type="randint", low=50, high=200),
            num_leaves=ParameterSpace(type="randint", low=15, high=63),
        )
        space = build_lightgbm_search_space(config)

        assert "n_estimators" in space
        assert "num_leaves" in space

    def test_build_catboost_search_space(self):
        """Test building CatBoost search space."""
        config = CatBoostSearchSpace(
            iterations=ParameterSpace(type="randint", low=50, high=200),
            depth=ParameterSpace(type="randint", low=4, high=10),
        )
        space = build_catboost_search_space(config)

        assert "iterations" in space
        assert "depth" in space

    def test_build_chemeleon_search_space(self):
        """Test building Chemeleon search space."""
        config = ChemeleonSearchSpace(
            ffn_hidden_dim=ParameterSpace(type="choice", values=[128, 256]),
            dropout=ParameterSpace(type="uniform", low=0.0, high=0.5),
        )
        space = build_chemeleon_search_space(config)

        assert "ffn_hidden_dim" in space
        assert "dropout" in space

    def test_build_fingerprint_search_space(self):
        """Test building fingerprint search space with prefix."""
        config = FingerprintSearchSpace(
            fp_type=ParameterSpace(type="choice", values=["morgan", "rdkit"]),
            morgan_radius=ParameterSpace(type="randint", low=2, high=4),
        )
        space = build_fingerprint_search_space(config)

        assert "fp_fp_type" in space
        assert "fp_morgan_radius" in space

    def test_build_unified_search_space(self):
        """Test building unified search space."""
        config = HPOSearchSpaceConfig(
            model_type="xgboost",
            xgboost=XGBoostSearchSpace(
                n_estimators=ParameterSpace(type="randint", low=50, high=200),
            ),
            fingerprint=FingerprintSearchSpace(
                fp_type=ParameterSpace(type="choice", values=["morgan"]),
            ),
        )
        space = build_search_space(config)

        assert "n_estimators" in space
        assert "fp_fp_type" in space

    def test_build_search_space_skips_none(self):
        """Test that None parameters are skipped."""
        config = XGBoostSearchSpace(
            n_estimators=ParameterSpace(type="randint", low=50, high=200),
            max_depth=None,  # Should be skipped
        )
        space = build_xgboost_search_space(config)

        assert "n_estimators" in space
        assert "max_depth" not in space
