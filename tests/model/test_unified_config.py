"""Tests for unified configuration schema.

This module tests the UnifiedModelConfig dataclass and model-type-aware
validation introduced in the config harmonization.
"""

import pytest
from omegaconf import OmegaConf

from admet.model.config import (
    CLASSICAL_MODEL_TYPES,
    MODEL_TYPES,
    PYTORCH_MODEL_TYPES,
    ConfigValidationError,
    CurriculumConfig,
    InterTaskAffinityConfig,
    JointSamplingConfig,
    TaskAffinityConfig,
    TaskOversamplingConfig,
    UnifiedDataConfig,
    UnifiedMlflowConfig,
    UnifiedModelConfig,
    UnifiedOptimizationConfig,
    get_structured_config_for_model_type,
    validate_model_config,
)


class TestUnifiedModelConfig:
    """Test UnifiedModelConfig schema."""

    def test_can_create_structured_config(self):
        """Test OmegaConf.structured() works with UnifiedModelConfig."""
        config = OmegaConf.structured(UnifiedModelConfig)
        assert config.model.type == "chemprop"

    def test_default_values(self):
        """Test default values are set correctly."""
        config = OmegaConf.structured(UnifiedModelConfig)

        # Model defaults
        assert config.model.type == "chemprop"
        assert config.model.chemprop.depth == 5

        # Data defaults
        assert config.data.smiles_col == "SMILES"
        assert config.data.target_cols == []

        # Optimization defaults
        assert config.optimization.max_epochs == 150
        assert config.optimization.batch_size == 32
        assert config.optimization.criterion == "MAE"

        # MLflow defaults
        assert config.mlflow.enabled is True
        assert config.mlflow.experiment_name == "admet"

        # Training strategy defaults
        assert config.joint_sampling.enabled is False
        assert config.task_affinity.enabled is False

    def test_can_merge_with_yaml(self, tmp_path):
        """Test merging with YAML file."""
        yaml_content = """
model:
  type: xgboost
  xgboost:
    n_estimators: 200
data:
  smiles_col: SMILES
  target_cols: [LogD, LogS]
optimization:
  max_epochs: 50
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        base = OmegaConf.structured(UnifiedModelConfig)
        override = OmegaConf.load(yaml_file)
        config = OmegaConf.merge(base, override)

        assert config.model.type == "xgboost"
        assert config.model.xgboost.n_estimators == 200
        assert config.data.target_cols == ["LogD", "LogS"]
        assert config.optimization.max_epochs == 50

    def test_all_model_types_accessible(self):
        """Test all model type params are accessible."""
        config = OmegaConf.structured(UnifiedModelConfig)

        # Chemprop
        assert config.model.chemprop.depth == 5
        assert config.model.chemprop.message_hidden_dim == 600

        # Chemeleon
        assert config.model.chemeleon.checkpoint_path == "auto"

        # XGBoost
        assert config.model.xgboost.n_estimators == 100

        # LightGBM
        assert config.model.lightgbm.n_estimators == 100

        # CatBoost
        assert config.model.catboost.iterations == 100

        # Fingerprint
        assert config.model.fingerprint.type == "morgan"


class TestTrainingStrategyConfigs:
    """Test training strategy configs are model-agnostic."""

    def test_joint_sampling_config(self):
        """Test JointSamplingConfig has expected fields."""
        config = JointSamplingConfig()
        assert config.enabled is False
        assert config.seed == 42
        assert isinstance(config.task_oversampling, TaskOversamplingConfig)
        assert isinstance(config.curriculum, CurriculumConfig)

    def test_curriculum_config(self):
        """Test CurriculumConfig has expected fields."""
        config = CurriculumConfig()
        assert config.enabled is False
        assert config.quality_col == "Quality"
        assert config.patience == 5
        assert config.qualities == ["high", "medium", "low"]

    def test_task_affinity_config(self):
        """Test TaskAffinityConfig has expected fields."""
        config = TaskAffinityConfig()
        assert config.enabled is False
        assert config.n_groups == 3

    def test_inter_task_affinity_config(self):
        """Test InterTaskAffinityConfig has expected fields."""
        config = InterTaskAffinityConfig()
        assert config.enabled is False
        assert config.compute_every_n_steps == 1


class TestConfigValidation:
    """Test model-type-aware validation."""

    def test_classical_with_curriculum_raises(self):
        """Classical models cannot use curriculum."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "xgboost"
        config.joint_sampling.enabled = True
        config.joint_sampling.curriculum.enabled = True

        with pytest.raises(ConfigValidationError, match="not supported for xgboost"):
            validate_model_config(config)

    def test_classical_with_task_oversampling_raises(self):
        """Classical models cannot use task oversampling."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "lightgbm"
        config.joint_sampling.enabled = True
        config.joint_sampling.task_oversampling.alpha = 0.5

        with pytest.raises(ConfigValidationError, match="not supported for lightgbm"):
            validate_model_config(config)

    def test_classical_with_inter_task_affinity_raises(self):
        """Classical models cannot use inter-task affinity."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "catboost"
        config.inter_task_affinity.enabled = True

        with pytest.raises(ConfigValidationError, match="not supported for catboost"):
            validate_model_config(config)

    def test_classical_with_task_affinity_raises(self):
        """Classical models cannot use task affinity grouping."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "xgboost"
        config.task_affinity.enabled = True

        with pytest.raises(ConfigValidationError, match="not supported for xgboost"):
            validate_model_config(config)

    def test_neural_with_curriculum_ok(self):
        """Neural models can use curriculum."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "chemeleon"
        config.joint_sampling.enabled = True
        config.joint_sampling.curriculum.enabled = True

        validate_model_config(config)  # Should not raise

    def test_chemprop_with_all_features_ok(self):
        """Chemprop can use all training features."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "chemprop"
        config.joint_sampling.enabled = True
        config.joint_sampling.curriculum.enabled = True
        config.joint_sampling.task_oversampling.alpha = 0.5
        config.task_affinity.enabled = True
        config.inter_task_affinity.enabled = True

        validate_model_config(config)  # Should not raise

    def test_classical_without_training_strategies_ok(self):
        """Classical models work without training strategies."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.model.type = "xgboost"
        config.joint_sampling.enabled = False

        validate_model_config(config)  # Should not raise


class TestGetStructuredConfigForModelType:
    """Test factory function for model-type configs."""

    def test_chemprop_config(self):
        """Test Chemprop config has correct defaults."""
        config = get_structured_config_for_model_type("chemprop")
        assert config.model.type == "chemprop"
        # Neural models keep training strategies enabled by default
        assert config.joint_sampling.enabled is False  # Not enabled by default

    def test_chemeleon_config(self):
        """Test Chemeleon config has correct defaults."""
        config = get_structured_config_for_model_type("chemeleon")
        assert config.model.type == "chemeleon"

    def test_classical_config_disables_neural_features(self):
        """Test classical model configs disable neural-only features."""
        for model_type in ("xgboost", "lightgbm", "catboost"):
            config = get_structured_config_for_model_type(model_type)
            assert config.model.type == model_type
            assert config.joint_sampling.enabled is False
            assert config.joint_sampling.task_oversampling.alpha == 0.0
            assert config.joint_sampling.curriculum.enabled is False
            assert config.task_affinity.enabled is False
            assert config.inter_task_affinity.enabled is False

    def test_unknown_model_type_raises(self):
        """Test unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_structured_config_for_model_type("unknown")


class TestUnifiedDataConfig:
    """Test UnifiedDataConfig fields."""

    def test_default_values(self):
        """Test default values."""
        config = UnifiedDataConfig()
        assert config.smiles_col == "SMILES"
        assert config.target_cols == []
        assert config.splits is None
        assert config.folds is None
        assert config.quality_col is None

    def test_quality_col_for_curriculum(self):
        """Test quality_col can be set for curriculum."""
        config = UnifiedDataConfig(quality_col="Quality")
        assert config.quality_col == "Quality"


class TestUnifiedMlflowConfig:
    """Test UnifiedMlflowConfig fields."""

    def test_enabled_field(self):
        """Test 'enabled' is the canonical field name."""
        config = UnifiedMlflowConfig()
        assert config.enabled is True

    def test_all_fields(self):
        """Test all MLflow config fields."""
        config = UnifiedMlflowConfig(
            enabled=False,
            tracking_uri="http://localhost:5000",
            experiment_name="test_exp",
            run_name="test_run",
        )
        assert config.enabled is False
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "test_exp"
        assert config.run_name == "test_run"


class TestUnifiedOptimizationConfig:
    """Test UnifiedOptimizationConfig fields."""

    def test_neural_network_fields(self):
        """Test neural network training fields."""
        config = UnifiedOptimizationConfig()
        assert config.init_lr == 1.0e-4
        assert config.max_lr == 1.0e-3
        assert config.final_lr == 1.0e-4
        assert config.warmup_epochs == 5
        assert config.patience == 15

    def test_common_fields(self):
        """Test fields common to all models."""
        config = UnifiedOptimizationConfig()
        assert config.seed == 42
        assert config.criterion == "MAE"
        assert config.batch_size == 32


class TestConfigMergePreservesUserOverrides:
    """Test that config merging preserves user values."""

    def test_merge_preserves_user_model_type(self):
        """Test user model type is preserved after merge."""
        base = get_structured_config_for_model_type("chemprop")
        user = OmegaConf.create({"model": {"type": "xgboost"}})
        merged = OmegaConf.merge(base, user)

        assert merged.model.type == "xgboost"

    def test_merge_preserves_user_target_cols(self):
        """Test user target_cols are preserved."""
        base = get_structured_config_for_model_type("chemprop")
        user = OmegaConf.create({"data": {"target_cols": ["LogD", "LogS"]}})
        merged = OmegaConf.merge(base, user)

        assert merged.data.target_cols == ["LogD", "LogS"]

    def test_merge_preserves_nested_model_params(self):
        """Test nested model params are preserved."""
        base = get_structured_config_for_model_type("chemprop")
        user = OmegaConf.create(
            {
                "model": {
                    "type": "chemprop",
                    "chemprop": {"depth": 6, "message_hidden_dim": 1000},
                }
            }
        )
        merged = OmegaConf.merge(base, user)

        assert merged.model.chemprop.depth == 6
        assert merged.model.chemprop.message_hidden_dim == 1000
        # Other defaults preserved
        assert merged.model.chemprop.dropout == 0.1


class TestEnsembleSection:
    """Test EnsembleSection configuration."""

    def test_default_values(self):
        """Test ensemble section default values."""
        config = OmegaConf.structured(UnifiedModelConfig)
        assert config.ensemble.enabled is False
        assert config.ensemble.n_models == 5
        assert config.ensemble.aggregation == "mean"
        assert config.ensemble.use_splits is True
        assert config.ensemble.splits is None
        assert config.ensemble.folds is None

    def test_can_enable_ensemble(self):
        """Test enabling ensemble mode."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.ensemble.enabled = True
        config.ensemble.n_models = 10
        assert config.ensemble.enabled is True
        assert config.ensemble.n_models == 10

    def test_split_fold_filtering(self):
        """Test split/fold filtering."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.ensemble.splits = [0, 1, 2]
        config.ensemble.folds = [0, 1]
        assert config.ensemble.splits == [0, 1, 2]
        assert config.ensemble.folds == [0, 1]


class TestRayConfig:
    """Test RayConfig configuration."""

    def test_default_values(self):
        """Test ray config default values."""
        config = OmegaConf.structured(UnifiedModelConfig)
        assert config.ray.max_parallel == 1
        assert config.ray.num_cpus is None
        assert config.ray.num_gpus is None

    def test_can_set_resources(self):
        """Test setting Ray resources."""
        config = OmegaConf.structured(UnifiedModelConfig)
        config.ray.max_parallel = 4
        config.ray.num_cpus = 8
        config.ray.num_gpus = 2
        assert config.ray.max_parallel == 4
        assert config.ray.num_cpus == 8
        assert config.ray.num_gpus == 2


class TestModelTypeConstants:
    """Test model type constants are consistent."""

    def test_all_model_types_defined(self):
        """Test MODEL_TYPES contains all expected types."""
        expected = {"chemprop", "chemeleon", "xgboost", "lightgbm", "catboost"}
        assert set(MODEL_TYPES) == expected

    def test_pytorch_classical_union_equals_all(self):
        """Test PYTORCH + CLASSICAL = ALL model types."""
        assert set(PYTORCH_MODEL_TYPES) | set(CLASSICAL_MODEL_TYPES) == set(MODEL_TYPES)

    def test_pytorch_classical_disjoint(self):
        """Test PYTORCH and CLASSICAL are disjoint sets."""
        assert set(PYTORCH_MODEL_TYPES) & set(CLASSICAL_MODEL_TYPES) == set()

    def test_pytorch_model_types(self):
        """Test PYTORCH_MODEL_TYPES are correct."""
        assert set(PYTORCH_MODEL_TYPES) == {"chemprop", "chemeleon"}

    def test_classical_model_types(self):
        """Test CLASSICAL_MODEL_TYPES are correct."""
        assert set(CLASSICAL_MODEL_TYPES) == {"xgboost", "lightgbm", "catboost"}
