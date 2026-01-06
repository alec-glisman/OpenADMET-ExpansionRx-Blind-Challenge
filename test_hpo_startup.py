#!/usr/bin/env python
"""Test script to validate HPO scripts start without errors.

This script tests that both chemprop and chemeleon HPO configurations
load properly and the training process begins without crashes.
"""

import subprocess
import sys
import time


def test_hpo_startup(config_path: str, model_name: str, timeout: int = 60) -> bool:
    """Test that an HPO script starts without errors.

    Parameters
    ----------
    config_path : str
        Path to the HPO config YAML file
    model_name : str
        Name of the model being tested (for display)
    timeout : int
        Number of seconds to run before terminating

    Returns
    -------
    bool
        True if the process started successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name} HPO startup...")
    print(f"Config: {config_path}")
    print(f"Timeout: {timeout}s")
    print(f"{'='*60}\n")

    cmd = ["admet", "model", "hpo", "-c", config_path]

    try:
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        # Check that we got a valid stdout
        if process.stdout is None:
            print("\n❌ Failed to capture process output")
            return False

        # Collect output for a few seconds
        start_time = time.time()
        output_lines = []
        error_detected = False

        while time.time() - start_time < timeout:
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
                print(line, end="")

                # Check for critical errors
                if any(err in line.lower() for err in ["error:", "traceback", "exception"]):
                    if "traceback" in line.lower() or "error:" in line.lower():
                        error_detected = True

            # Check if process died
            if process.poll() is not None:
                # Process ended
                remaining = process.stdout.read()
                if remaining:
                    output_lines.append(remaining)
                    print(remaining, end="")

                returncode = process.returncode
                if returncode != 0:
                    print(f"\n❌ Process exited with code {returncode}")
                    return False
                break

            time.sleep(0.1)

        # Kill the process if still running
        if process.poll() is None:
            print(f"\n⏱️  Timeout reached ({timeout}s) - terminating process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        # Check results
        if error_detected:
            print(f"\n❌ {model_name} HPO failed - errors detected in output")
            return False
        else:
            print(f"\n✅ {model_name} HPO started successfully!")
            return True

    except Exception as e:
        print(f"\n❌ {model_name} HPO failed with exception: {e}")
        return False


def main():
    """Run startup tests for both chemprop and chemeleon HPO."""
    results = {}

    # Test Chemprop
    results["chemprop"] = test_hpo_startup("configs/1-hpo-single/hpo_chemprop.yaml", "Chemprop", timeout=60)

    # Test Chemeleon
    results["chemeleon"] = test_hpo_startup("configs/1-hpo-single/hpo_chemeleon.yaml", "CheMeleon", timeout=60)

    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model:15} {status}")
    print(f"{'='*60}\n")

    # Exit with appropriate code
    if all(results.values()):
        print("✅ All HPO scripts validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some HPO scripts failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
