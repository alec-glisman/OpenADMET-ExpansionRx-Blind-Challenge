"""Unit tests for HPO configuration module."""

from omegaconf import OmegaConf

from admet.model.chemprop.hpo_config import (
    ASHAConfig,
    HPOConfig,
    ParameterSpace,
    ResourceConfig,
    SearchSpaceConfig,
    TransferLearningConfig,
)


class TestParameterSpace:
    """Tests for ParameterSpace dataclass."""

    def test_uniform_parameter(self) -> None:
        """Test uniform distribution parameter."""
        param = ParameterSpace(type="uniform", low=0.0, high=1.0)
        assert param.type == "uniform"
        assert param.low == 0.0
        assert param.high == 1.0
        assert param.values is None

    def test_loguniform_parameter(self) -> None:
        """Test log-uniform distribution parameter."""
        param = ParameterSpace(type="loguniform", low=1e-5, high=1e-2)
        assert param.type == "loguniform"
        assert param.low == 1e-5
        assert param.high == 1e-2

    def test_choice_parameter(self) -> None:
        """Test choice distribution parameter."""
        param = ParameterSpace(type="choice", values=[32, 64, 128])
        assert param.type == "choice"
        assert param.values == [32, 64, 128]
        assert param.low is None

    def test_conditional_parameter(self) -> None:
        """Test conditional parameter configuration."""
        param = ParameterSpace(
            type="choice",
            values=[2, 4, 8],
            conditional_on="ffn_type",
            conditional_values=["moe"],
        )
        assert param.conditional_on == "ffn_type"
        assert param.conditional_values == ["moe"]


class TestSearchSpaceConfig:
    """Tests for SearchSpaceConfig dataclass."""

    def test_empty_search_space(self) -> None:
        """Test default empty search space."""
        config = SearchSpaceConfig()
        assert config.learning_rate is None
        assert config.dropout is None
        assert config.ffn_type is None

    def test_partial_search_space(self) -> None:
        """Test search space with some parameters."""
        config = SearchSpaceConfig(
            learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
            dropout=ParameterSpace(type="uniform", low=0.0, high=0.4),
        )
        assert config.learning_rate is not None
        assert config.learning_rate.type == "loguniform"
        assert config.dropout is not None
        assert config.ffn_hidden_dim is None


class TestASHAConfig:
    """Tests for ASHAConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default ASHA configuration."""
        config = ASHAConfig()
        assert config.metric == "val_mae"
        assert config.mode == "min"
        assert config.max_t == 100
        assert config.grace_period == 15
        assert config.reduction_factor == 3
        assert config.brackets == 1

    def test_custom_values(self) -> None:
        """Test custom ASHA configuration."""
        config = ASHAConfig(
            metric="val_loss",
            mode="min",
            max_t=100,
            grace_period=5,
            reduction_factor=2,
        )
        assert config.metric == "val_loss"
        assert config.max_t == 100
        assert config.grace_period == 5


class TestResourceConfig:
    """Tests for ResourceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default resource configuration."""
        config = ResourceConfig()
        assert config.num_samples == 500
        assert config.cpus_per_trial == 4
        assert config.gpus_per_trial == 0.25
        assert config.max_concurrent_trials is None

    def test_fractional_gpu(self) -> None:
        """Test fractional GPU allocation."""
        config = ResourceConfig(gpus_per_trial=0.25)
        # 0.25 GPU = 4 concurrent trials per GPU
        assert config.gpus_per_trial == 0.25


class TestTransferLearningConfig:
    """Tests for TransferLearningConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default transfer learning configuration."""
        config = TransferLearningConfig()
        assert config.top_k == 5
        assert config.full_epochs == 150
        assert config.ensemble_size == 5

    def test_custom_values(self) -> None:
        """Test custom transfer learning configuration."""
        config = TransferLearningConfig(
            top_k=3,
            full_epochs=200,
            ensemble_size=10,
        )
        assert config.top_k == 3
        assert config.full_epochs == 200


class TestHPOConfig:
    """Tests for HPOConfig dataclass."""

    def test_minimal_config(self) -> None:
        """Test minimal HPO configuration."""
        config = HPOConfig(
            experiment_name="test_hpo",
            data_path="train.csv",
            target_columns=["target1"],
        )
        assert config.experiment_name == "test_hpo"
        assert config.data_path == "train.csv"
        assert config.smiles_column == "smiles"
        assert config.target_columns == ["target1"]

    def test_full_config(self) -> None:
        """Test full HPO configuration."""
        config = HPOConfig(
            experiment_name="full_hpo",
            data_path="train.csv",
            val_data_path="validation.csv",
            smiles_column="SMILES",
            target_columns=["logD", "solubility"],
            output_dir="outputs/hpo",
            seed=123,
            search_space=SearchSpaceConfig(
                learning_rate=ParameterSpace(type="loguniform", low=1e-5, high=1e-2),
            ),
            asha=ASHAConfig(max_t=100),
            resources=ResourceConfig(num_samples=30),
        )
        assert config.val_data_path == "validation.csv"
        assert config.seed == 123
        assert config.asha.max_t == 100
        assert config.resources.num_samples == 30

    def test_omegaconf_compatibility(self) -> None:
        """Test that config can be used with OmegaConf."""
        config = HPOConfig(
            experiment_name="omega_test",
            data_path="data.csv",
            target_columns=["target"],
        )
        structured = OmegaConf.structured(config)
        # OmegaConf.structured returns a DictConfig; verify it's usable
        assert structured is not None

        # Should be able to convert to container
        container = OmegaConf.to_container(structured)
        assert isinstance(container, dict)
        assert container["experiment_name"] == "omega_test"
