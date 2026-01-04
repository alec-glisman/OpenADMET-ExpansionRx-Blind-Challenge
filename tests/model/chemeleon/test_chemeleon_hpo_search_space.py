"""Tests for CheMeleon HPO search space builder with conditional parameters.

This module tests the conditional parameter handling in CheMeleon HPO:
- freeze_encoder conditional: unfreeze_encoder_epoch, unfreeze_encoder_lr_multiplier
- ffn_type conditional: n_experts (MoE), trunk_depth/trunk_hidden_dim (branched)
"""

from __future__ import annotations

import pytest
from ray import tune

from admet.model.chemeleon.hpo_config import ChemeleonSearchSpaceConfig, ParameterSpace
from admet.model.chemeleon.hpo_search_space import _build_parameter_space, build_chemeleon_search_space


class TestBuildParameterSpace:
    """Tests for _build_parameter_space function."""

    def test_uniform_distribution(self) -> None:
        """Test building uniform distribution."""
        param = ParameterSpace(type="uniform", low=0.0, high=1.0)
        result = _build_parameter_space(param)
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
        param = ParameterSpace(type="quniform", low=100, high=500, q=100)
        result = _build_parameter_space(param)
        assert result is not None

    def test_randint_distribution(self) -> None:
        """Test building randint distribution."""
        param = ParameterSpace(type="randint", low=1, high=10)
        result = _build_parameter_space(param)
        assert result is not None

    def test_qrandint_distribution(self) -> None:
        """Test building quantized randint distribution."""
        param = ParameterSpace(type="qrandint", low=2, high=8, q=2)
        result = _build_parameter_space(param)
        assert result is not None

    def test_invalid_type_raises(self) -> None:
        """Test that invalid type raises ValueError."""
        param = ParameterSpace(type="invalid")
        with pytest.raises(ValueError, match="Unknown parameter type"):
            _build_parameter_space(param)


class TestBuildChemeleonSearchSpace:
    """Tests for build_chemeleon_search_space function."""

    def test_empty_config(self) -> None:
        """Test building search space from empty config."""
        config = ChemeleonSearchSpaceConfig()
        space = build_chemeleon_search_space(config)
        assert isinstance(space, dict)
        assert len(space) == 0

    def test_simple_parameters(self) -> None:
        """Test building search space with simple (non-conditional) parameters."""
        config = ChemeleonSearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
            dropout=ParameterSpace(type="uniform", low=0.0, high=0.3),
            batch_size=ParameterSpace(type="choice", values=[32, 64, 128]),
            weight_decay=ParameterSpace(type="loguniform", low=1e-6, high=0.1),
        )
        space = build_chemeleon_search_space(config)
        assert "learning_rate" in space
        assert "dropout" in space
        assert "batch_size" in space
        assert "weight_decay" in space

    def test_ffn_type_parameter(self) -> None:
        """Test that ffn_type is included in search space."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts", "branched"]),
        )
        space = build_chemeleon_search_space(config)
        assert "ffn_type" in space


class TestConditionalMoEParameters:
    """Tests for MoE-specific conditional parameters (n_experts)."""

    def test_conditional_n_experts_is_sample_from(self) -> None:
        """Test that conditional n_experts uses tune.sample_from."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=8,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
        )
        space = build_chemeleon_search_space(config)
        assert "ffn_type" in space
        assert "n_experts" in space
        assert isinstance(space["n_experts"], tune.search.sample.Function)

    def test_n_experts_sampled_when_moe(self) -> None:
        """Test n_experts returns value when ffn_type is mixture_of_experts."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["mixture_of_experts"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=8,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["n_experts"]
        config_dict = {"ffn_type": "mixture_of_experts"}
        result = sample_fn.func(config_dict)

        assert result is not None
        assert 2 <= result <= 8

    def test_n_experts_none_when_not_moe(self) -> None:
        """Test n_experts returns None when ffn_type is not mixture_of_experts."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=8,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["n_experts"]
        config_dict = {"ffn_type": "regression"}
        result = sample_fn.func(config_dict)

        assert result is None

    def test_non_conditional_n_experts(self) -> None:
        """Test non-conditional n_experts is not wrapped in sample_from."""
        config = ChemeleonSearchSpaceConfig(
            n_experts=ParameterSpace(type="randint", low=2, high=8),
        )
        space = build_chemeleon_search_space(config)
        assert "n_experts" in space
        assert not isinstance(space["n_experts"], tune.search.sample.Function)


class TestConditionalBranchedParameters:
    """Tests for Branched FFN conditional parameters (trunk_depth, trunk_hidden_dim)."""

    def test_conditional_trunk_depth_is_sample_from(self) -> None:
        """Test that conditional trunk_depth uses tune.sample_from."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "branched"]),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)
        assert "trunk_depth" in space
        assert isinstance(space["trunk_depth"], tune.search.sample.Function)

    def test_trunk_depth_sampled_when_branched(self) -> None:
        """Test trunk_depth returns value when ffn_type is branched."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["branched"]),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["trunk_depth"]
        config_dict = {"ffn_type": "branched"}
        result = sample_fn.func(config_dict)

        assert result is not None
        assert 0 <= result <= 5

    def test_trunk_depth_none_when_not_branched(self) -> None:
        """Test trunk_depth returns None when ffn_type is not branched."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "branched"]),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["trunk_depth"]
        config_dict = {"ffn_type": "regression"}
        result = sample_fn.func(config_dict)

        assert result is None

    def test_conditional_trunk_hidden_dim_choice(self) -> None:
        """Test trunk_hidden_dim with choice values when branched."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["branched"]),
            trunk_hidden_dim=ParameterSpace(
                type="choice",
                values=[100, 200, 300, 400],
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["trunk_hidden_dim"]
        config_dict = {"ffn_type": "branched"}
        result = sample_fn.func(config_dict)

        assert result in [100, 200, 300, 400]

    def test_trunk_hidden_dim_none_when_not_branched(self) -> None:
        """Test trunk_hidden_dim returns None when ffn_type is not branched."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "branched"]),
            trunk_hidden_dim=ParameterSpace(
                type="choice",
                values=[100, 200, 300],
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["trunk_hidden_dim"]
        config_dict = {"ffn_type": "regression"}
        result = sample_fn.func(config_dict)

        assert result is None


class TestConditionalFreezeEncoderParameters:
    """Tests for freeze_encoder conditional parameters (unfreeze_encoder_epoch)."""

    def test_conditional_unfreeze_epoch_is_sample_from(self) -> None:
        """Test that conditional unfreeze_encoder_epoch uses tune.sample_from."""
        config = ChemeleonSearchSpaceConfig(
            freeze_encoder=ParameterSpace(type="choice", values=[True, False]),
            unfreeze_encoder_epoch=ParameterSpace(
                type="randint",
                low=10,
                high=50,
                conditional_on="freeze_encoder",
                conditional_values=[True],
            ),
        )
        space = build_chemeleon_search_space(config)
        assert "unfreeze_encoder_epoch" in space
        assert isinstance(space["unfreeze_encoder_epoch"], tune.search.sample.Function)

    def test_unfreeze_epoch_sampled_when_frozen(self) -> None:
        """Test unfreeze_encoder_epoch returns value when freeze_encoder=True."""
        config = ChemeleonSearchSpaceConfig(
            freeze_encoder=ParameterSpace(type="choice", values=[True]),
            unfreeze_encoder_epoch=ParameterSpace(
                type="randint",
                low=10,
                high=50,
                conditional_on="freeze_encoder",
                conditional_values=[True],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["unfreeze_encoder_epoch"]
        config_dict = {"freeze_encoder": True}
        result = sample_fn.func(config_dict)

        assert result is not None
        assert 10 <= result <= 50

    def test_unfreeze_epoch_none_when_not_frozen(self) -> None:
        """Test unfreeze_encoder_epoch returns None when freeze_encoder=False."""
        config = ChemeleonSearchSpaceConfig(
            freeze_encoder=ParameterSpace(type="choice", values=[True, False]),
            unfreeze_encoder_epoch=ParameterSpace(
                type="randint",
                low=10,
                high=50,
                conditional_on="freeze_encoder",
                conditional_values=[True],
            ),
        )
        space = build_chemeleon_search_space(config)

        sample_fn = space["unfreeze_encoder_epoch"]
        config_dict = {"freeze_encoder": False}
        result = sample_fn.func(config_dict)

        assert result is None


class TestSearchSpaceFromYAMLConfig:
    """Tests validating search space built from YAML-like configs."""

    def test_full_chemeleon_hpo_config(self) -> None:
        """Test building search space matching hpo_chemeleon.yaml structure."""
        config = ChemeleonSearchSpaceConfig(
            freeze_encoder=ParameterSpace(type="choice", values=[True, False]),
            unfreeze_encoder_epoch=ParameterSpace(
                type="randint",
                low=10,
                high=50,
                conditional_on="freeze_encoder",
                conditional_values=[True],
            ),
            unfreeze_encoder_lr_multiplier=ParameterSpace(
                type="loguniform",
                low=0.01,
                high=0.5,
            ),
            learning_rate=ParameterSpace(type="loguniform", low=0.0001, high=0.03),
            lr_warmup_ratio=ParameterSpace(type="loguniform", low=0.1, high=10.0),
            lr_final_ratio=ParameterSpace(type="loguniform", low=0.001, high=1.0),
            weight_decay=ParameterSpace(type="loguniform", low=1e-6, high=0.1),
            dropout=ParameterSpace(type="uniform", low=0.0, high=0.3),
            batch_size=ParameterSpace(type="choice", values=[32, 64, 128, 256]),
            batch_norm=ParameterSpace(type="choice", values=[True, False]),
            ffn_type=ParameterSpace(
                type="choice",
                values=["regression", "mixture_of_experts", "branched"],
            ),
            ffn_num_layers=ParameterSpace(type="randint", low=0, high=5),
            ffn_hidden_dim=ParameterSpace(type="quniform", low=200, high=1200, q=100),
            n_experts=ParameterSpace(
                type="qrandint",
                low=2,
                high=8,
                q=2,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
            trunk_hidden_dim=ParameterSpace(
                type="quniform",
                low=100,
                high=1200,
                q=100,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        # Verify all expected keys are present
        expected_simple = [
            "freeze_encoder",
            "unfreeze_encoder_lr_multiplier",
            "learning_rate",
            "lr_warmup_ratio",
            "lr_final_ratio",
            "weight_decay",
            "dropout",
            "batch_size",
            "batch_norm",
            "ffn_type",
            "ffn_num_layers",
            "ffn_hidden_dim",
        ]
        for key in expected_simple:
            assert key in space, f"Missing key: {key}"

        # Verify conditional parameters use sample_from
        conditional_params = [
            "unfreeze_encoder_epoch",
            "n_experts",
            "trunk_depth",
            "trunk_hidden_dim",
        ]
        for key in conditional_params:
            assert key in space, f"Missing conditional key: {key}"
            assert isinstance(space[key], tune.search.sample.Function), f"{key} should be sample_from"

    def test_moe_config_scenario(self) -> None:
        """Test MoE scenario: n_experts is sampled, branched params are None."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts", "branched"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=8,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        moe_config = {"ffn_type": "mixture_of_experts"}
        n_experts_result = space["n_experts"].func(moe_config)
        trunk_depth_result = space["trunk_depth"].func(moe_config)

        assert n_experts_result is not None and 2 <= n_experts_result <= 8
        assert trunk_depth_result is None

    def test_branched_config_scenario(self) -> None:
        """Test Branched scenario: trunk params sampled, n_experts is None."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts", "branched"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=8,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        branched_config = {"ffn_type": "branched"}
        n_experts_result = space["n_experts"].func(branched_config)
        trunk_depth_result = space["trunk_depth"].func(branched_config)

        assert n_experts_result is None
        assert trunk_depth_result is not None and 0 <= trunk_depth_result <= 5

    def test_regression_config_scenario(self) -> None:
        """Test Regression scenario: all conditional params are None."""
        config = ChemeleonSearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["regression", "mixture_of_experts", "branched"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=8,
                conditional_on="ffn_type",
                conditional_values=["mixture_of_experts"],
            ),
            trunk_depth=ParameterSpace(
                type="randint",
                low=0,
                high=5,
                conditional_on="ffn_type",
                conditional_values=["branched"],
            ),
        )
        space = build_chemeleon_search_space(config)

        regression_config = {"ffn_type": "regression"}
        n_experts_result = space["n_experts"].func(regression_config)
        trunk_depth_result = space["trunk_depth"].func(regression_config)

        assert n_experts_result is None
        assert trunk_depth_result is None

    def test_frozen_vs_unfrozen_encoder_scenario(self) -> None:
        """Test encoder freezing scenarios."""
        config = ChemeleonSearchSpaceConfig(
            freeze_encoder=ParameterSpace(type="choice", values=[True, False]),
            unfreeze_encoder_epoch=ParameterSpace(
                type="randint",
                low=10,
                high=50,
                conditional_on="freeze_encoder",
                conditional_values=[True],
            ),
        )
        space = build_chemeleon_search_space(config)

        # When frozen, unfreeze_epoch is sampled
        frozen_config = {"freeze_encoder": True}
        unfreeze_result = space["unfreeze_encoder_epoch"].func(frozen_config)
        assert unfreeze_result is not None

        # When not frozen, unfreeze_epoch is None
        unfrozen_config = {"freeze_encoder": False}
        unfreeze_result = space["unfreeze_encoder_epoch"].func(unfrozen_config)
        assert unfreeze_result is None
