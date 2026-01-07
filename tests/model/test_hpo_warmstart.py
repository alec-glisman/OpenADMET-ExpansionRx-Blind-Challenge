"""Tests for HPO warmstart functionality."""

import pytest

from admet.model.chemprop.hpo_config import SearchAlgorithmConfig


class TestSearchAlgorithmConfigExtension:
    """Test SearchAlgorithmConfig with persistence fields."""

    def test_config_defaults(self):
        """Test SearchAlgorithmConfig defaults."""
        config = SearchAlgorithmConfig()

        assert config.type == "optuna"
        assert config.seed == 42
        assert config.n_initial_points == 20

        # New persistence fields
        assert config.persist_study is False
        assert config.study_name is None
        assert config.storage_dir is None
        assert config.warmstart_from is None
        assert config.warmstart_n_trials == 10

    def test_config_with_persistence_enabled(self):
        """Test SearchAlgorithmConfig with persistence enabled."""
        config = SearchAlgorithmConfig(
            persist_study=True,
            study_name="test_study",
            storage_dir="/tmp/optuna",
            warmstart_from="previous_study",
            warmstart_n_trials=15,
        )

        assert config.persist_study is True
        assert config.study_name == "test_study"
        assert config.storage_dir == "/tmp/optuna"
        assert config.warmstart_from == "previous_study"
        assert config.warmstart_n_trials == 15


class TestWarmstartLogic:
    """Test warmstart implementation logic."""

    @pytest.mark.skip(reason="Requires Optuna database - integration test")
    def test_study_persistence_creates_database(self, hpo_config_with_persistence):
        """Test that persistent study creates SQLite database."""
        # Would verify:
        # - Run HPO with persist_study=True
        # - Check that studies.db exists
        # - Verify study can be loaded from database
        pass

    @pytest.mark.skip(reason="Requires Optuna database - integration test")
    def test_warmstart_enqueues_top_trials(self, hpo_config_with_warmstart):
        """Test warmstart enqueues top trials from previous study."""
        # Would verify:
        # - Run initial HPO study
        # - Run warmstart study referencing first
        # - Verify first N trials match top N from previous study
        pass

    @pytest.mark.skip(reason="Requires Optuna database - integration test")
    def test_warmstart_handles_missing_study(self, hpo_config_with_invalid_warmstart):
        """Test warmstart gracefully handles missing study."""
        # Would verify:
        # - Config references non-existent study
        # - HPO continues without warmstart (logs warning)
        # - No crash or error
        pass


class TestStudyMetadata:
    """Test study metadata logging."""

    @pytest.mark.skip(reason="Requires full HPO run - integration test")
    def test_study_metadata_saved(self, completed_hpo_run):
        """Test study metadata JSON is created."""
        # Would verify:
        # - study_metadata.json exists
        # - Contains study_name, storage_dir, n_trials, best_metric
        # - Contains warmstart_from if applicable
        pass
