"""Unit tests for HPO search space builder module."""

import pytest
from ray import tune

from admet.model.chemprop.hpo_config import ParameterSpace, SearchSpaceConfig
from admet.model.chemprop.hpo_search_space import (
    _build_parameter_space,
    build_search_space,
    get_default_search_space,
)


class TestBuildParameterSpace:
    """Tests for _build_parameter_space function."""

    def test_uniform_distribution(self) -> None:
        """Test building uniform distribution."""
        param = ParameterSpace(type="uniform", low=0.0, high=1.0)
        result = _build_parameter_space(param)
        # Ray Tune returns a sampler object
        assert result is not None

    def test_loguniform_distribution(self) -> None:
        """Test building log-uniform distribution."""
        param = ParameterSpace(type="loguniform", low=1e-5, high=1e-2)
        result = _build_parameter_space(param)
        assert result is not None

    def test_choice_distribution(self) -> None:
        """Test building choice distribution."""
        param = ParameterSpace(type="choice", values=[32, 64, 128])
        result = _build_parameter_space(param)
        assert result is not None

    def test_quniform_distribution(self) -> None:
        """Test building quantized uniform distribution."""
        param = ParameterSpace(type="quniform", low=1, high=10, q=1)
        result = _build_parameter_space(param)
        assert result is not None

    def test_invalid_type_raises(self) -> None:
        """Test that invalid type raises ValueError."""
        param = ParameterSpace(type="invalid")
        with pytest.raises(ValueError, match="Unknown parameter type"):
            _build_parameter_space(param)

    def test_uniform_missing_bounds_raises(self) -> None:
        """Test that uniform without bounds raises ValueError."""
        param = ParameterSpace(type="uniform", low=0.0)  # missing high
        with pytest.raises(ValueError, match="requires 'low' and 'high'"):
            _build_parameter_space(param)

    def test_choice_missing_values_raises(self) -> None:
        """Test that choice without values raises ValueError."""
        param = ParameterSpace(type="choice")
        with pytest.raises(ValueError, match="requires 'values'"):
            _build_parameter_space(param)


class TestBuildSearchSpace:
    """Tests for build_search_space function."""

    def test_empty_config(self) -> None:
        """Test building search space from empty config."""
        config = SearchSpaceConfig()
        space = build_search_space(config)
        assert isinstance(space, dict)
        assert len(space) == 0

    def test_simple_parameters(self) -> None:
        """Test building search space with simple parameters."""
        config = SearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
            dropout=ParameterSpace(type="uniform", low=0.0, high=0.4),
            batch_size=ParameterSpace(type="choice", values=[32, 64, 128]),
        )
        space = build_search_space(config)
        assert "learning_rate" in space
        assert "dropout" in space
        assert "batch_size" in space

    def test_ffn_type_parameter(self) -> None:
        """Test that ffn_type is included in search space."""
        config = SearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["mlp", "moe", "branched"]),
        )
        space = build_search_space(config)
        assert "ffn_type" in space

    def test_conditional_n_experts(self) -> None:
        """Test conditional n_experts parameter."""
        config = SearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["mlp", "moe"]),
            n_experts=ParameterSpace(
                type="choice",
                values=[2, 4, 8],
                conditional_on="ffn_type",
                conditional_values=["moe"],
            ),
        )
        space = build_search_space(config)
        assert "ffn_type" in space
        assert "n_experts" in space
        # n_experts should be a sample_from for conditional sampling
        assert isinstance(space["n_experts"], tune.search.sample.Function)

    def test_non_conditional_n_experts(self) -> None:
        """Test non-conditional n_experts parameter."""
        config = SearchSpaceConfig(
            n_experts=ParameterSpace(type="choice", values=[2, 4, 8]),
        )
        space = build_search_space(config)
        assert "n_experts" in space

    def test_target_weights_per_column(self) -> None:
        """Test that target_weights creates per-target parameters."""
        config = SearchSpaceConfig(
            target_weights=ParameterSpace(type="uniform", low=0.05, high=50.0),
        )
        target_columns = ["LogD", "Log KSOL", "Log HLM CLint"]
        space = build_search_space(config, target_columns=target_columns)

        # Should have one weight parameter per target
        assert "target_weight_LogD" in space
        assert "target_weight_Log_KSOL" in space
        assert "target_weight_Log_HLM_CLint" in space

    def test_target_weights_safe_names(self) -> None:
        """Test that target weight names handle special characters."""
        config = SearchSpaceConfig(
            target_weights=ParameterSpace(type="uniform", low=0.05, high=50.0),
        )
        target_columns = ["Log Caco-2 Permeability Papp A>B"]
        space = build_search_space(config, target_columns=target_columns)

        # > should be replaced with gt
        assert "target_weight_Log_Caco-2_Permeability_Papp_AgtB" in space

    def test_target_weights_without_columns(self) -> None:
        """Test that target_weights without columns creates no params."""
        config = SearchSpaceConfig(
            target_weights=ParameterSpace(type="uniform", low=0.05, high=50.0),
        )
        # No target_columns passed
        space = build_search_space(config)
        # Should not have any target_weight params
        assert not any(k.startswith("target_weight_") for k in space)


class TestGetDefaultSearchSpace:
    """Tests for get_default_search_space function."""

    def test_returns_search_space_config(self) -> None:
        """Test that function returns SearchSpaceConfig."""
        config = get_default_search_space()
        assert isinstance(config, SearchSpaceConfig)

    def test_has_expected_parameters(self) -> None:
        """Test that default config has expected parameters."""
        config = get_default_search_space()
        assert config.learning_rate is not None
        assert config.dropout is not None
        assert config.depth is not None
        assert config.message_hidden_dim is not None  # Replaced hidden_dim
        assert config.ffn_type is not None
        assert config.batch_size is not None
        # New LR schedule parameters
        assert config.lr_warmup_ratio is not None
        assert config.lr_final_ratio is not None

    def test_n_experts_is_conditional(self) -> None:
        """Test that n_experts is conditional on ffn_type."""
        config = get_default_search_space()
        assert config.n_experts is not None
        assert config.n_experts.conditional_on == "ffn_type"
        assert "moe" in (config.n_experts.conditional_values or [])

    def test_has_target_weights(self) -> None:
        """Test that default config includes target_weights."""
        config = get_default_search_space()
        assert config.target_weights is not None
        assert config.target_weights.type == "uniform"
        assert config.target_weights.low == 0.05
        assert config.target_weights.high == 50.0
