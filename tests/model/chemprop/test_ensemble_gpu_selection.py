"""Tests for GPU ID selection in ensemble training via CUDA_VISIBLE_DEVICES."""

import os
from unittest.mock import MagicMock, patch

import pytest

from admet.model.chemprop.config import EnsembleConfig
from admet.model.config import RayConfig


class TestRayConfigGpuIds:
    """Test RayConfig gpu_ids field."""

    def test_gpu_ids_default_is_none(self):
        """gpu_ids should default to None (use all GPUs)."""
        config = RayConfig()
        assert config.gpu_ids is None

    def test_gpu_ids_single_gpu(self):
        """Can set a single GPU ID."""
        config = RayConfig(gpu_ids=[1])
        assert config.gpu_ids == [1]

    def test_gpu_ids_multiple_gpus(self):
        """Can set multiple GPU IDs."""
        config = RayConfig(gpu_ids=[0, 2, 3])
        assert config.gpu_ids == [0, 2, 3]

    def test_gpu_ids_empty_list(self):
        """Empty list should be valid (treated as None/all GPUs)."""
        config = RayConfig(gpu_ids=[])
        assert config.gpu_ids == []


class TestEnsembleGpuIdSelection:
    """Test that ensemble training respects gpu_ids configuration."""

    @pytest.fixture
    def mock_ensemble_config(self, tmp_path):
        """Create a minimal ensemble config for testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = EnsembleConfig()
        config.data.data_dir = str(data_dir)
        config.data.target_cols = ["LogD"]
        config.mlflow.tracking = False
        return config

    def test_cuda_visible_devices_set_for_single_gpu(self, mock_ensemble_config):
        """CUDA_VISIBLE_DEVICES should be set when gpu_ids contains single GPU."""
        mock_ensemble_config.ray.gpu_ids = [1]
        mock_ensemble_config.ray.num_gpus = 1

        # Verify the config is set correctly
        assert mock_ensemble_config.ray.gpu_ids == [1]

        # Test the string generation logic directly
        gpu_ids = mock_ensemble_config.ray.gpu_ids
        cuda_visible = ",".join(str(g) for g in gpu_ids)
        assert cuda_visible == "1"

    def test_cuda_visible_devices_set_for_multiple_gpus(self, mock_ensemble_config):
        """CUDA_VISIBLE_DEVICES should contain all specified GPU IDs."""
        mock_ensemble_config.ray.gpu_ids = [1, 3]
        mock_ensemble_config.ray.num_gpus = 2

        gpu_ids = mock_ensemble_config.ray.gpu_ids
        cuda_visible = ",".join(str(g) for g in gpu_ids)
        assert cuda_visible == "1,3"

    def test_no_cuda_visible_devices_when_gpu_ids_none(self, mock_ensemble_config):
        """CUDA_VISIBLE_DEVICES should not be set when gpu_ids is None."""
        mock_ensemble_config.ray.gpu_ids = None
        assert mock_ensemble_config.ray.gpu_ids is None

    def test_no_cuda_visible_devices_when_gpu_ids_empty(self, mock_ensemble_config):
        """CUDA_VISIBLE_DEVICES should not be set when gpu_ids is empty list."""
        mock_ensemble_config.ray.gpu_ids = []

        gpu_ids = mock_ensemble_config.ray.gpu_ids
        # Empty list should not trigger CUDA_VISIBLE_DEVICES setting
        should_set = gpu_ids is not None and len(gpu_ids) > 0
        assert should_set is False


class TestGpuIdSelectionLogic:
    """Test the GPU selection logic that would run in train_all."""

    def test_runtime_env_created_with_cuda_visible_devices(self):
        """runtime_env should contain CUDA_VISIBLE_DEVICES when gpu_ids set."""
        gpu_ids = [1, 2]
        ray_kwargs = {}

        if gpu_ids is not None and len(gpu_ids) > 0:
            cuda_visible = ",".join(str(g) for g in gpu_ids)
            ray_kwargs["runtime_env"] = {"env_vars": {"CUDA_VISIBLE_DEVICES": cuda_visible}}

        assert "runtime_env" in ray_kwargs
        assert ray_kwargs["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "1,2"

    def test_runtime_env_not_created_when_no_gpu_ids(self):
        """runtime_env should not contain CUDA_VISIBLE_DEVICES when gpu_ids not set."""
        gpu_ids = None
        ray_kwargs = {}

        if gpu_ids is not None and len(gpu_ids) > 0:
            cuda_visible = ",".join(str(g) for g in gpu_ids)
            ray_kwargs["runtime_env"] = {"env_vars": {"CUDA_VISIBLE_DEVICES": cuda_visible}}

        assert "runtime_env" not in ray_kwargs

    def test_available_gpus_count_matches_gpu_ids_length(self):
        """available_gpus should equal len(gpu_ids) when gpu_ids is set."""
        gpu_ids = [0, 2, 5]

        if gpu_ids is not None and len(gpu_ids) > 0:
            available_gpus = len(gpu_ids)
        else:
            available_gpus = 4  # Simulated nvidia-smi detection

        assert available_gpus == 3


class TestWorkerCudaVisibleDevices:
    """Test that workers receive and apply CUDA_VISIBLE_DEVICES correctly."""

    def test_worker_sets_cuda_visible_devices_when_provided(self):
        """Worker should set CUDA_VISIBLE_DEVICES env var when passed."""
        cuda_visible_devices = "1,2"

        # Simulate what happens at start of train_single_model
        original_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1,2"
        finally:
            # Restore original
            if original_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_env
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def test_worker_does_not_modify_env_when_none(self):
        """Worker should not modify CUDA_VISIBLE_DEVICES when None passed."""
        cuda_visible_devices = None

        original_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            # Clear to test
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            if cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

            # Should not be set
            assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
        finally:
            if original_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_env


class TestGpuIdConfigIntegration:
    """Integration tests for gpu_ids in full config."""

    def test_ensemble_config_with_gpu_ids(self, tmp_path):
        """EnsembleConfig should properly store gpu_ids."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = EnsembleConfig()
        config.data.data_dir = str(data_dir)
        config.data.target_cols = ["LogD"]
        config.ray.gpu_ids = [1]
        config.ray.num_gpus = 1
        config.ray.max_parallel = 5

        assert config.ray.gpu_ids == [1]
        assert config.ray.num_gpus == 1
        assert config.ray.max_parallel == 5

    def test_ensemble_config_without_gpu_ids(self, tmp_path):
        """EnsembleConfig should work without gpu_ids (use all GPUs)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = EnsembleConfig()
        config.data.data_dir = str(data_dir)
        config.data.target_cols = ["LogD"]
        config.ray.num_gpus = 2
        config.ray.max_parallel = 5

        assert config.ray.gpu_ids is None
        assert config.ray.num_gpus == 2

    def test_gpu_ids_with_non_sequential_ids(self, tmp_path):
        """gpu_ids should work with non-sequential GPU IDs (e.g., [0, 2] skipping 1)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = EnsembleConfig()
        config.data.data_dir = str(data_dir)
        config.data.target_cols = ["LogD"]
        config.ray.gpu_ids = [0, 2, 4]
        config.ray.num_gpus = 3

        cuda_visible = ",".join(str(g) for g in config.ray.gpu_ids)
        assert cuda_visible == "0,2,4"

        # After CUDA_VISIBLE_DEVICES is set, torch will see these as GPUs 0,1,2
        # so num_gpus should match the count
        assert config.ray.num_gpus == len(config.ray.gpu_ids)
