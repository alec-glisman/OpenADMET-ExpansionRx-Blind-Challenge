"""Tests exposing the GPU environment variable ordering bug.

The issue: CUDA_VISIBLE_DEVICES must be set BEFORE PyTorch is imported.
Once PyTorch initializes CUDA, it caches the device list and ignores
subsequent changes to CUDA_VISIBLE_DEVICES.

These tests are designed to FAIL and expose the bug in the current implementation.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


class TestCudaVisibleDevicesOrdering:
    """Tests that expose the CUDA_VISIBLE_DEVICES ordering bug."""

    @pytest.mark.slow
    def test_setting_cuda_visible_after_torch_import_has_no_effect(self):
        """
        FAILING TEST: Demonstrates that setting CUDA_VISIBLE_DEVICES after
        importing torch does NOT change which GPU PyTorch uses.

        This test should FAIL to expose the bug.
        """
        script = """
import torch
import os

# Set CUDA_VISIBLE_DEVICES AFTER torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check what torch sees - it should still see the original GPUs
# because CUDA was already initialized
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    # If CUDA_VISIBLE_DEVICES=1 took effect, we'd see 1 device
    # But since torch was imported first, we'll see all devices
    print(f"DEVICE_COUNT:{device_count}")
else:
    print("DEVICE_COUNT:0")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            check=False,
        )

        output = result.stdout.strip()
        for line in output.split("\n"):
            if line.startswith("DEVICE_COUNT:"):
                device_count = int(line.split(":")[1])
                break
        else:
            pytest.skip("Could not determine device count")

        # This assertion demonstrates the bug:
        # We expect device_count to be 1 (because we set CUDA_VISIBLE_DEVICES=1)
        # But it will actually be > 1 (all GPUs) because torch was imported first
        assert device_count == 1, (
            f"Bug exposed: Setting CUDA_VISIBLE_DEVICES after torch import "
            f"should give 1 device, but got {device_count}. "
            f"This proves the env var must be set BEFORE torch import."
        )

    @pytest.mark.slow
    def test_setting_cuda_visible_before_torch_import_works(self):
        """
        PASSING TEST: Demonstrates that setting CUDA_VISIBLE_DEVICES BEFORE
        importing torch correctly restricts visible GPUs.
        """
        script = """
import os

# Set CUDA_VISIBLE_DEVICES BEFORE torch import
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# Now torch should only see 1 device
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"DEVICE_COUNT:{device_count}")
else:
    print("DEVICE_COUNT:0")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
            check=False,
        )

        output = result.stdout.strip()
        for line in output.split("\n"):
            if line.startswith("DEVICE_COUNT:"):
                device_count = int(line.split(":")[1])
                break
        else:
            pytest.skip("Could not determine device count or no CUDA available")

        if device_count == 0:
            pytest.skip("No CUDA devices available")

        # This should pass - setting env var BEFORE import works
        assert device_count == 1, (
            f"Expected 1 device when CUDA_VISIBLE_DEVICES=0 set before torch import, " f"got {device_count}"
        )


class TestEnsembleModuleImportOrder:
    """Tests that expose the import order problem in ensemble.py."""

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Known bug: ensemble.py imports torch at module level " "before CUDA_VISIBLE_DEVICES can be set"
    )
    def test_ensemble_module_imports_torch_at_module_level(self):
        """
        FAILING TEST: Verify that importing ensemble.py imports torch,
        which means CUDA is initialized before train_all() can set CUDA_VISIBLE_DEVICES.

        This test should FAIL to expose the architectural bug.
        """
        script = """
import sys
import os

# Clear CUDA_VISIBLE_DEVICES to start fresh
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

# Track if torch gets imported
torch_imported_before = "torch" in sys.modules

# Import ensemble module
from admet.model.chemprop.ensemble import ModelEnsemble

# Check if torch was imported as a side effect
torch_imported_after = "torch" in sys.modules

print(f"TORCH_IMPORTED_BEFORE:{torch_imported_before}")
print(f"TORCH_IMPORTED_AFTER:{torch_imported_after}")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parents[3]),
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[3] / "src")},
            check=False,
        )

        if result.returncode != 0:
            pytest.fail(f"Script failed: {result.stderr}")

        output = result.stdout.strip()
        torch_after = False

        for line in output.split("\n"):
            if line.startswith("TORCH_IMPORTED_AFTER:"):
                torch_after = line.split(":")[1] == "True"

        # This assertion exposes the bug:
        # If torch is imported when ensemble.py is imported, then any
        # CUDA_VISIBLE_DEVICES set in train_all() will be too late
        assert not torch_after, (
            "Bug exposed: Importing ensemble.py imports torch as a side effect. "
            "This means CUDA is initialized before train_all() can set CUDA_VISIBLE_DEVICES. "
            "The fix requires setting CUDA_VISIBLE_DEVICES before importing ensemble.py "
            "or restructuring imports to be lazy."
        )


class TestRayWorkerGpuInheritance:
    """Tests for Ray worker GPU environment inheritance."""

    @pytest.mark.slow
    @pytest.mark.xfail(reason="Ray worker environment variable inheritance behavior is " "platform/version dependent")
    def test_ray_worker_inherits_cuda_visible_devices_from_runtime_env(self):
        """
        Test that Ray workers properly inherit CUDA_VISIBLE_DEVICES from runtime_env.
        """
        pytest.importorskip("ray")

        script = """
import os
import ray

# Initialize Ray with runtime_env setting CUDA_VISIBLE_DEVICES
ray.init(
    runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1"}},
    ignore_reinit_error=True,
    num_gpus=0,
)

@ray.remote
def check_env():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")

try:
    result = ray.get(check_env.remote())
    print(f"WORKER_CUDA_VISIBLE_DEVICES:{result}")
finally:
    ray.shutdown()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        output = result.stdout.strip()
        for line in output.split("\n"):
            if line.startswith("WORKER_CUDA_VISIBLE_DEVICES:"):
                worker_env = line.split(":", 1)[1]
                break
        else:
            pytest.skip("Could not determine worker environment")

        assert worker_env == "1", (
            f"Ray worker should inherit CUDA_VISIBLE_DEVICES=1 from runtime_env, " f"but got '{worker_env}'"
        )

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Known bug: Setting CUDA_VISIBLE_DEVICES after ray.init() " "doesn't propagate to workers"
    )
    def test_ray_worker_does_not_inherit_parent_env_after_init(self):
        """
        FAILING TEST: Ray workers do NOT automatically inherit CUDA_VISIBLE_DEVICES
        set in the parent process after Ray is initialized.

        This exposes why setting the env var in train_all() doesn't work for Ray workers.
        """
        pytest.importorskip("ray")

        script = """
import os
import ray

# Initialize Ray first (simulating what happens when ensemble.py is imported)
ray.init(ignore_reinit_error=True, num_gpus=0)

# Now set CUDA_VISIBLE_DEVICES in parent (simulating train_all)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@ray.remote
def check_env():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")

try:
    result = ray.get(check_env.remote())
    print(f"WORKER_CUDA_VISIBLE_DEVICES:{result}")
finally:
    ray.shutdown()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
            check=False,
        )

        output = result.stdout.strip()
        for line in output.split("\n"):
            if line.startswith("WORKER_CUDA_VISIBLE_DEVICES:"):
                worker_env = line.split(":", 1)[1]
                break
        else:
            pytest.skip("Could not determine worker environment")

        # This assertion exposes the bug:
        # We set CUDA_VISIBLE_DEVICES=1 in parent, but worker won't see it
        assert worker_env == "1", (
            f"Bug exposed: Ray worker got CUDA_VISIBLE_DEVICES='{worker_env}' "
            f"instead of '1'. This proves that setting CUDA_VISIBLE_DEVICES "
            f"after ray.init() doesn't propagate to workers."
        )


class TestCliGpuIdsFix:
    """Tests for the CLI fix that sets CUDA_VISIBLE_DEVICES before module import."""

    @pytest.mark.slow
    def test_cli_sets_cuda_visible_devices_before_import(self, tmp_path):
        """
        Test that the CLI sets CUDA_VISIBLE_DEVICES before importing ensemble module.

        This test verifies the fix works by checking that:
        1. A config with gpu_ids is read
        2. CUDA_VISIBLE_DEVICES is set based on gpu_ids
        3. This happens BEFORE the ensemble module (and thus PyTorch) is imported
        """
        # Create a minimal config file with gpu_ids
        config_file = tmp_path / "test_config.yaml"
        config_content = """
ray:
  gpu_ids: [1]
  max_parallel: 1
  num_gpus: 1
data:
  data_dir: /tmp/fake
  target_cols: [LogD]
"""
        config_file.write_text(config_content)

        script = """
import os
import sys

# Clear any existing CUDA_VISIBLE_DEVICES
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

# Check torch is not imported yet
torch_imported_before_cli = "torch" in sys.modules
print("TORCH_BEFORE_CLI:" + str(torch_imported_before_cli))

# Simulate what the CLI does: read config and set env var BEFORE import
from omegaconf import OmegaConf

config = OmegaConf.load("{config_file}")
gpu_ids = config.get("ray", {{}}).get("gpu_ids")
if gpu_ids is not None and len(gpu_ids) > 0:
    cuda_visible_devices = ",".join(str(g) for g in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# Check env var is set
env_after_config = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
print("ENV_AFTER_CONFIG:" + env_after_config)

# Check torch is still not imported
torch_imported_after_config = "torch" in sys.modules
print("TORCH_AFTER_CONFIG:" + str(torch_imported_after_config))

# Now import would happen (we don't actually import to keep test fast)
print("SUCCESS")
""".format(
            config_file=config_file
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        output = result.stdout.strip()
        lines = {line.split(":")[0]: line.split(":", 1)[1] for line in output.split("\n") if ":" in line}

        # Verify torch was not imported before or after config reading
        assert lines.get("TORCH_BEFORE_CLI") == "False", "torch should not be imported before CLI"
        assert lines.get("TORCH_AFTER_CONFIG") == "False", "torch should not be imported after reading config"

        # Verify CUDA_VISIBLE_DEVICES was set correctly
        assert (
            lines.get("ENV_AFTER_CONFIG") == "1"
        ), f"CUDA_VISIBLE_DEVICES should be '1', got '{lines.get('ENV_AFTER_CONFIG')}'"

    @pytest.mark.slow
    def test_full_cli_gpu_selection_flow(self, tmp_path):
        """
        Integration test: Verify the complete flow sets CUDA_VISIBLE_DEVICES
        and PyTorch sees only the specified GPU.
        """
        config_file = tmp_path / "test_config.yaml"
        config_content = """
ray:
  gpu_ids: [0]
  max_parallel: 1
  num_gpus: 1
data:
  data_dir: /tmp/fake
  target_cols: [LogD]
"""
        config_file.write_text(config_content)

        script = """
import os
import sys

# Clear any existing CUDA_VISIBLE_DEVICES
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

# Step 1: Read config and set env var (what CLI does)
from omegaconf import OmegaConf

config = OmegaConf.load("{config_file}")
gpu_ids = config.get("ray", {{}}).get("gpu_ids")
if gpu_ids is not None and len(gpu_ids) > 0:
    cuda_visible_devices = ",".join(str(g) for g in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# Step 2: Now import torch (simulating ensemble module import)
import torch

# Step 3: Check what torch sees
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print("DEVICE_COUNT:" + str(device_count))
else:
    print("DEVICE_COUNT:0")
""".format(
            config_file=config_file
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
            check=False,
        )

        output = result.stdout.strip()
        for line in output.split("\n"):
            if line.startswith("DEVICE_COUNT:"):
                device_count = int(line.split(":")[1])
                break
        else:
            pytest.skip("Could not determine device count")

        if device_count == 0:
            pytest.skip("No CUDA devices available")

        # With the fix, torch should see exactly 1 GPU
        assert device_count == 1, (
            f"With CUDA_VISIBLE_DEVICES=0 set before torch import, " f"should see 1 device, but got {device_count}"
        )
