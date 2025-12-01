"""
Unit tests for admet.model.chemprop.config module.

Tests configuration dataclasses including CurriculumConfig, ChempropConfig,
EnsembleConfig, and their integration with OmegaConf.
"""

import pytest
from omegaconf import OmegaConf

from admet.model.chemprop.config import (
    ChempropConfig,
    CurriculumConfig,
    DataConfig,
    EnsembleConfig,
    EnsembleDataConfig,
    MlflowConfig,
    ModelConfig,
    OptimizationConfig,
)


class TestCurriculumConfig:
    """Tests for CurriculumConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default curriculum configuration values."""
        config = CurriculumConfig()
        assert config.enabled is False
        assert config.quality_col == "Quality"
        assert config.qualities == ["high", "medium", "low"]
        assert config.patience == 5
        assert config.seed == 42

    def test_enabled_curriculum(self) -> None:
        """Test enabled curriculum with custom values."""
        config = CurriculumConfig(
            enabled=True,
            quality_col="DataQuality",
            qualities=["excellent", "good", "fair", "poor"],
            patience=10,
            seed=123,
        )
        assert config.enabled is True
        assert config.quality_col == "DataQuality"
        assert len(config.qualities) == 4
        assert config.patience == 10
        assert config.seed == 123

    def test_two_quality_levels(self) -> None:
        """Test curriculum with only two quality levels."""
        config = CurriculumConfig(
            enabled=True,
            qualities=["high", "low"],
        )
        assert len(config.qualities) == 2
        assert config.qualities[0] == "high"
        assert config.qualities[1] == "low"

    def test_omegaconf_structured(self) -> None:
        """Test CurriculumConfig with OmegaConf structured config."""
        config = OmegaConf.structured(CurriculumConfig)
        assert config.enabled is False
        assert config.seed == 42

    def test_omegaconf_merge(self) -> None:
        """Test merging YAML-like config with CurriculumConfig."""
        base = OmegaConf.structured(CurriculumConfig)
        override = OmegaConf.create(
            {
                "enabled": True,
                "quality_col": "QualityLabel",
                "patience": 3,
            }
        )
        merged = OmegaConf.merge(base, override)
        assert merged.enabled is True
        assert merged.quality_col == "QualityLabel"
        assert merged.patience == 3
        # Defaults should be preserved
        assert merged.seed == 42


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default data configuration values."""
        config = DataConfig()
        assert config.smiles_col == "SMILES"
        assert config.target_cols == []
        assert config.target_weights == []
        assert config.output_dir is None

    def test_with_target_cols(self) -> None:
        """Test data config with target columns."""
        config = DataConfig(
            data_dir="/path/to/data",
            smiles_col="smiles",
            target_cols=["LogD", "Log KSOL"],
            target_weights=[1.0, 2.0],
        )
        assert config.data_dir == "/path/to/data"
        assert len(config.target_cols) == 2
        assert len(config.target_weights) == 2


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default model configuration values."""
        config = ModelConfig()
        assert config.depth == 5
        assert config.message_hidden_dim == 600
        assert config.num_layers == 2
        assert config.hidden_dim == 600
        assert config.dropout == 0.1
        assert config.batch_norm is True
        assert config.ffn_type == "regression"

    def test_moe_config(self) -> None:
        """Test mixture of experts configuration."""
        config = ModelConfig(
            ffn_type="mixture_of_experts",
            n_experts=8,
        )
        assert config.ffn_type == "mixture_of_experts"
        assert config.n_experts == 8

    def test_branched_config(self) -> None:
        """Test branched FFN configuration."""
        config = ModelConfig(
            ffn_type="branched",
            trunk_n_layers=3,
            trunk_hidden_dim=512,
        )
        assert config.ffn_type == "branched"
        assert config.trunk_n_layers == 3
        assert config.trunk_hidden_dim == 512


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default optimization configuration values."""
        config = OptimizationConfig()
        assert config.criterion == "MAE"
        assert config.init_lr == 1.0e-4
        assert config.max_lr == 1.0e-3
        assert config.final_lr == 1.0e-4
        assert config.warmup_epochs == 5
        assert config.patience == 15
        assert config.max_epochs == 150
        assert config.batch_size == 32
        assert config.seed == 12345

    def test_custom_learning_rates(self) -> None:
        """Test custom learning rate schedule."""
        config = OptimizationConfig(
            init_lr=1e-5,
            max_lr=1e-2,
            final_lr=1e-6,
            warmup_epochs=10,
        )
        assert config.init_lr == 1e-5
        assert config.max_lr == 1e-2


class TestMlflowConfig:
    """Tests for MlflowConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default MLflow configuration values."""
        config = MlflowConfig()
        assert config.tracking is True
        assert config.tracking_uri is None
        assert config.experiment_name == "chemprop"
        assert config.nested is False

    def test_nested_run(self) -> None:
        """Test nested run configuration."""
        config = MlflowConfig(
            parent_run_id="abc123",
            nested=True,
        )
        assert config.parent_run_id == "abc123"
        assert config.nested is True


class TestChempropConfig:
    """Tests for ChempropConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default ChempropConfig has all sub-configs."""
        config = ChempropConfig()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.mlflow, MlflowConfig)
        assert isinstance(config.curriculum, CurriculumConfig)

    def test_curriculum_enabled(self) -> None:
        """Test ChempropConfig with curriculum enabled."""
        config = ChempropConfig(
            curriculum=CurriculumConfig(enabled=True, patience=3),
        )
        assert config.curriculum.enabled is True
        assert config.curriculum.patience == 3

    def test_full_config_from_omegaconf(self) -> None:
        """Test loading full config from OmegaConf."""
        yaml_str = """
        data:
          data_dir: /path/to/data
          smiles_col: SMILES
          target_cols:
            - LogD
            - Log KSOL
        model:
          depth: 4
          hidden_dim: 512
        optimization:
          max_epochs: 100
          patience: 10
        mlflow:
          experiment_name: test_experiment
        curriculum:
          enabled: true
          patience: 5
        """
        base = OmegaConf.structured(ChempropConfig)
        override = OmegaConf.create(yaml_str)
        config = OmegaConf.merge(base, override)

        assert config.data.data_dir == "/path/to/data"
        assert len(config.data.target_cols) == 2
        assert config.model.depth == 4
        assert config.optimization.max_epochs == 100
        assert config.curriculum.enabled is True


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default EnsembleConfig has all sub-configs."""
        config = EnsembleConfig()
        assert isinstance(config.data, EnsembleDataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.mlflow, MlflowConfig)
        assert isinstance(config.curriculum, CurriculumConfig)
        assert config.max_parallel == 1

    def test_ensemble_with_curriculum(self) -> None:
        """Test EnsembleConfig with curriculum learning."""
        config = EnsembleConfig(
            curriculum=CurriculumConfig(enabled=True),
            max_parallel=4,
        )
        assert config.curriculum.enabled is True
        assert config.max_parallel == 4

    def test_splits_and_folds_filtering(self) -> None:
        """Test ensemble data config with split/fold filtering."""
        data_config = EnsembleDataConfig(
            data_dir="/path/to/splits",
            splits=[0, 1, 2],
            folds=[0, 1, 2, 3, 4],
        )
        assert data_config.splits == [0, 1, 2]
        assert data_config.folds == [0, 1, 2, 3, 4]
