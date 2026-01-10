"""Unit tests for HPO search space refinement module."""

import json
import math
from pathlib import Path

import pytest

from admet.model.chemprop.hpo_config import ParameterSpace, SearchSpaceConfig
from admet.model.hpo_range_refinement import (
    RefinementConfig,
    _compute_refined_range,
    analyze_parameter,
    generate_refined_phase3_config,
    load_top_configs,
    refine_search_space,
)


class TestLoadTopConfigs:
    """Tests for load_top_configs function."""

    def test_load_from_top_k_json(self, tmp_path: Path) -> None:
        """Test loading configs from top_k_configs.json."""
        configs = [
            {"learning_rate": 0.001, "depth": 3, "_rank": 1},
            {"learning_rate": 0.002, "depth": 4, "_rank": 2},
            {"learning_rate": 0.003, "depth": 5, "_rank": 3},
        ]

        json_path = tmp_path / "top_k_configs.json"
        with open(json_path, "w") as f:
            json.dump(configs, f)

        loaded = load_top_configs(tmp_path, top_k=2)
        assert len(loaded) == 2
        assert loaded[0]["learning_rate"] == 0.001
        assert loaded[1]["learning_rate"] == 0.002

    def test_load_from_top_k_json_limits_to_top_k(self, tmp_path: Path) -> None:
        """Test that top_k parameter limits results."""
        configs = [{"lr": i / 100} for i in range(10)]

        json_path = tmp_path / "top_k_configs.json"
        with open(json_path, "w") as f:
            json.dump(configs, f)

        loaded = load_top_configs(tmp_path, top_k=3)
        assert len(loaded) == 3

    def test_load_no_configs_raises(self, tmp_path: Path) -> None:
        """Test that missing config files raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_top_configs(tmp_path)


class TestComputeRefinedRange:
    """Tests for _compute_refined_range function."""

    def test_empty_values_returns_none(self) -> None:
        """Test that empty values after NaN filtering returns None."""
        result = _compute_refined_range(
            [],
            original_type="uniform",
            margin_factor=0.0,
            use_percentiles=False,
        )
        assert result is None

    def test_all_nan_values_returns_none(self) -> None:
        """Test that all-NaN values returns None."""
        result = _compute_refined_range(
            [float("nan"), float("nan")],
            original_type="uniform",
            margin_factor=0.0,
            use_percentiles=False,
        )
        assert result is None

    def test_uniform_range_no_margin(self) -> None:
        """Test uniform range computation without margin."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = _compute_refined_range(
            values,
            original_type="uniform",
            margin_factor=0.0,
            use_percentiles=False,
        )
        assert low == 1.0
        assert high == 5.0

    def test_uniform_range_with_margin(self) -> None:
        """Test uniform range computation with margin."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = _compute_refined_range(
            values,
            original_type="uniform",
            margin_factor=0.25,
            use_percentiles=False,
        )
        # Range is 4.0, margin is 1.0, so [0.0, 6.0]
        assert low == 0.0
        assert high == 6.0

    def test_loguniform_range_with_margin(self) -> None:
        """Test loguniform range computation (margin in log space)."""
        values = [0.001, 0.01, 0.1]
        low, high = _compute_refined_range(
            values,
            original_type="loguniform",
            margin_factor=0.0,
            use_percentiles=False,
        )
        assert math.isclose(low, 0.001, rel_tol=1e-9)
        assert math.isclose(high, 0.1, rel_tol=1e-9)

    def test_loguniform_margin_expands_in_log_space(self) -> None:
        """Test that loguniform margin is applied in log space."""
        values = [0.01, 0.1]  # log range = ln(0.1) - ln(0.01) ≈ 2.3
        low, high = _compute_refined_range(
            values,
            original_type="loguniform",
            margin_factor=0.5,
            use_percentiles=False,
        )
        # Margin factor 0.5 should expand range by 50% on each side in log space
        assert low < 0.01
        assert high > 0.1
        # Verify log-scale expansion
        log_range = math.log(0.1) - math.log(0.01)
        expected_log_low = math.log(0.01) - 0.5 * log_range
        expected_log_high = math.log(0.1) + 0.5 * log_range
        assert math.isclose(math.log(low), expected_log_low, rel_tol=1e-5)
        assert math.isclose(math.log(high), expected_log_high, rel_tol=1e-5)

    def test_percentile_range(self) -> None:
        """Test percentile-based range computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # 100 is outlier
        low, high = _compute_refined_range(
            values,
            original_type="uniform",
            margin_factor=0.0,
            use_percentiles=True,
            percentile_low=20,
            percentile_high=80,
        )
        # Percentile should exclude the outlier
        assert low < 100.0
        assert high < 100.0


class TestAnalyzeParameter:
    """Tests for analyze_parameter function."""

    def test_continuous_parameter(self) -> None:
        """Test analysis of continuous parameter."""
        configs = [
            {"learning_rate": 0.001},
            {"learning_rate": 0.005},
            {"learning_rate": 0.01},
        ]
        original = ParameterSpace(type="loguniform", low=1e-5, high=0.1)

        refined = analyze_parameter(
            param_name="learning_rate",
            configs=configs,
            original_param=original,
            margin_factor=0.0,
            use_percentiles=False,
        )

        assert refined.param_name == "learning_rate"
        assert refined.original_type == "loguniform"
        assert len(refined.observed_values) == 3
        assert math.isclose(refined.refined_low, 0.001, rel_tol=1e-9)
        assert math.isclose(refined.refined_high, 0.01, rel_tol=1e-9)

    def test_choice_parameter_multiple_values(self) -> None:
        """Test analysis of choice parameter with multiple observed values."""
        configs = [
            {"ffn_type": "mlp"},
            {"ffn_type": "moe"},
            {"ffn_type": "mlp"},
            {"ffn_type": "branched"},
        ]
        original = ParameterSpace(type="choice", values=["mlp", "moe", "branched"])

        refined = analyze_parameter(
            param_name="ffn_type",
            configs=configs,
            original_param=original,
        )

        assert refined.original_type == "choice"
        assert refined.fixed_value is None
        assert set(refined.refined_values) == {"mlp", "moe", "branched"}

    def test_choice_parameter_single_value_fixed(self) -> None:
        """Test that single observed choice value is fixed."""
        configs = [
            {"ffn_type": "mlp"},
            {"ffn_type": "mlp"},
            {"ffn_type": "mlp"},
        ]
        original = ParameterSpace(type="choice", values=["mlp", "moe", "branched"])

        refined = analyze_parameter(
            param_name="ffn_type",
            configs=configs,
            original_param=original,
        )

        assert refined.fixed_value == "mlp"
        assert refined.refined_values == ["mlp"]

    def test_missing_parameter_returns_empty(self) -> None:
        """Test that missing parameter returns empty observed_values."""
        configs = [{"other_param": 1}, {"other_param": 2}]
        original = ParameterSpace(type="uniform", low=0.0, high=1.0)

        refined = analyze_parameter(
            param_name="learning_rate",
            configs=configs,
            original_param=original,
        )

        assert len(refined.observed_values) == 0


class TestRefineSearchSpace:
    """Tests for refine_search_space function."""

    def test_refinement_disabled_returns_base(self) -> None:
        """Test that disabled refinement returns base config unchanged."""
        base = SearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=0.1),
        )
        refinement = RefinementConfig(enabled=False)

        result = refine_search_space(base, refinement)
        assert result.learning_rate == base.learning_rate

    def test_missing_previous_dir_returns_base(self) -> None:
        """Test that missing previous_phase_dir returns base config."""
        base = SearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=0.1),
        )
        refinement = RefinementConfig(enabled=True, previous_phase_dir=None)

        result = refine_search_space(base, refinement)
        assert result.learning_rate == base.learning_rate

    def test_refinement_narrows_continuous_params(self, tmp_path: Path) -> None:
        """Test that refinement narrows continuous parameter ranges."""
        # Create mock phase 2 results
        configs = [
            {"learning_rate": 0.001, "depth": 3},
            {"learning_rate": 0.002, "depth": 4},
            {"learning_rate": 0.003, "depth": 5},
        ]
        json_path = tmp_path / "top_k_configs.json"
        with open(json_path, "w") as f:
            json.dump(configs, f)

        base = SearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=0.1),
            depth=ParameterSpace(type="randint", low=1, high=10),
        )
        refinement = RefinementConfig(
            enabled=True,
            previous_phase_dir=str(tmp_path),
            top_k=10,
            margin_factor=0.0,
            use_percentiles=False,
        )

        result = refine_search_space(base, refinement)

        # Learning rate should be narrowed
        assert result.learning_rate is not None
        assert result.learning_rate.low >= 0.001
        assert result.learning_rate.high <= 0.003 * 1.1  # Small tolerance

        # Depth should be narrowed
        assert result.depth is not None
        assert result.depth.low == 3
        assert result.depth.high == 5

    def test_refinement_preserves_conditional_params(self, tmp_path: Path) -> None:
        """Test that conditional parameters are preserved after refinement."""
        configs = [
            {"n_experts": 4, "ffn_type": "moe"},
            {"n_experts": 6, "ffn_type": "moe"},
        ]
        json_path = tmp_path / "top_k_configs.json"
        with open(json_path, "w") as f:
            json.dump(configs, f)

        base = SearchSpaceConfig(
            ffn_type=ParameterSpace(type="choice", values=["mlp", "moe", "branched"]),
            n_experts=ParameterSpace(
                type="randint",
                low=2,
                high=10,
                conditional_on="ffn_type",
                conditional_values=["moe"],
            ),
        )
        refinement = RefinementConfig(
            enabled=True,
            previous_phase_dir=str(tmp_path),
            margin_factor=0.0,
            use_percentiles=False,
        )

        result = refine_search_space(base, refinement)

        # Conditional attributes should be preserved
        assert result.n_experts is not None
        assert result.n_experts.conditional_on == "ffn_type"
        assert result.n_experts.conditional_values == ["moe"]


class TestGenerateRefinedPhase3Config:
    """Tests for generate_refined_phase3_config function."""

    def test_generates_refined_config(self, tmp_path: Path) -> None:
        """Test that refined config is generated from phase 2 results."""
        # Create phase 2 output
        configs = [
            {"learning_rate": 0.002, "depth": 4, "batch_size": 64},
            {"learning_rate": 0.003, "depth": 5, "batch_size": 128},
            {"learning_rate": 0.001, "depth": 4, "batch_size": 64},
        ]
        phase2_dir = tmp_path / "phase2"
        phase2_dir.mkdir()
        with open(phase2_dir / "top_k_configs.json", "w") as f:
            json.dump(configs, f)

        # Create phase 3 template
        import yaml

        template = {
            "experiment_name": "phase3_test",
            "search_space": {
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 0.1},
                "depth": {"type": "randint", "low": 1, "high": 10},
                "batch_size": {"type": "choice", "values": [32, 64, 128, 256]},
            },
        }
        template_path = tmp_path / "phase3_template.yaml"
        with open(template_path, "w") as f:
            yaml.dump(template, f)

        # Generate refined config
        output_path = tmp_path / "phase3_refined.yaml"
        result = generate_refined_phase3_config(
            phase2_dir=phase2_dir,
            phase3_template_path=template_path,
            output_path=output_path,
            top_k=10,
            margin_factor=0.0,
        )

        # Verify result structure
        assert "search_space" in result
        assert "learning_rate" in result["search_space"]
        assert "_refinement_metadata" in result

        # Verify output file was created
        assert output_path.exists()

    def test_refines_choice_params(self, tmp_path: Path) -> None:
        """Test that choice parameters are narrowed to observed values."""
        configs = [
            {"batch_size": 64},
            {"batch_size": 128},
            {"batch_size": 64},
        ]
        phase2_dir = tmp_path / "phase2"
        phase2_dir.mkdir()
        with open(phase2_dir / "top_k_configs.json", "w") as f:
            json.dump(configs, f)

        import yaml

        template = {
            "search_space": {
                "batch_size": {"type": "choice", "values": [32, 64, 128, 256]},
            },
        }
        template_path = tmp_path / "template.yaml"
        with open(template_path, "w") as f:
            yaml.dump(template, f)

        result = generate_refined_phase3_config(
            phase2_dir=phase2_dir,
            phase3_template_path=template_path,
        )

        # Should only include observed batch sizes
        batch_sizes = result["search_space"]["batch_size"]["values"]
        assert set(batch_sizes) == {64, 128}
