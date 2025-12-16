"""CLI wrappers for model-related commands (training, ensemble, HPO).

These are thin wrappers that delegate to the existing module-level
`main()` functions so users can call `admet model train ...` instead of
`python -m admet.model.chemprop.model ...`.
"""

from __future__ import annotations

import importlib
import sys
from typing import List, Optional

import typer

model_app = typer.Typer(name="model", help="Model training and HPO commands")


def _run_module_main(module_name: str, args: List[str]) -> None:
    """Run module's main() with given argv (temporarily replaces sys.argv)."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [module_name] + args
        module = importlib.import_module(module_name)
        if not hasattr(module, "main"):
            raise RuntimeError(f"Module {module_name} has no main() function")
        module.main()
    finally:
        sys.argv = old_argv


@model_app.command("train")
def train(config: str = typer.Option(..., "--config", "-c", help="YAML config path")) -> None:
    """Train a single model using a Chemprop config YAML.

    Example:
        admet model train --config configs/0-experiment/chemprop.yaml
    """
    _run_module_main("admet.model.chemprop.model", ["--config", config])


@model_app.command("ensemble")
def ensemble(
    config: str = typer.Option(..., "--config", "-c", help="Ensemble config YAML"),
    max_parallel: Optional[int] = typer.Option(None, "--max-parallel", help="Max parallel models"),
) -> None:
    """Train an ensemble using a Chemprop ensemble config.

    Example:
        admet model ensemble --config configs/0-experiment/ensemble_chemprop_production.yaml --max-parallel 2
    """
    args = ["--config", config]
    if max_parallel is not None:
        args += ["--max-parallel", str(max_parallel)]
    _run_module_main("admet.model.chemprop.ensemble", args)


@model_app.command("hpo")
def hpo(
    config: str = typer.Option(..., "--config", "-c", help="HPO config YAML"),
    num_samples: Optional[int] = typer.Option(None, "--num-samples", help="Number of HPO trials"),
) -> None:
    """Run hyperparameter optimization (HPO) using a Chemprop HPO config.

    Example:
        admet model hpo --config configs/1-hpo-single/hpo_chemprop.yaml --num-samples 50
    """
    args = ["--config", config]
    if num_samples is not None:
        args += ["--num-samples", str(num_samples)]
    _run_module_main("admet.model.chemprop.hpo", args)
