"""
Shared FFN factory for Chemprop and CheMeleon models
=====================================================

This module provides a unified factory function for creating FFN (Feed-Forward Network)
predictors that can be used by both Chemprop and CheMeleon models.

Supported FFN architectures:
- ``regression``: Standard multi-task regression FFN (default)
- ``mixture_of_experts``: Mixture of experts regression FFN with gating network
- ``branched``: Branched FFN with shared trunk and task-specific heads
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chemprop import nn
from chemprop.nn import RegressionFFN

from admet.model.chemprop.ffn import BranchedFFN, MixtureOfExpertsRegressionFFN

if TYPE_CHECKING:
    from chemprop.nn.metrics import ChempropMetric
    from chemprop.nn.transforms import UnscaleTransform
    from torch import Tensor


def create_ffn_predictor(
    ffn_type: str,
    input_dim: int,
    n_tasks: int,
    hidden_dim: int = 300,
    n_layers: int = 2,
    dropout: float = 0.0,
    n_experts: int | None = None,
    trunk_n_layers: int | None = None,
    trunk_hidden_dim: int | None = None,
    task_groups: list[list[int]] | None = None,
    criterion: ChempropMetric | None = None,
    task_weights: Tensor | None = None,
    output_transform: UnscaleTransform | None = None,
) -> nn.Predictor:
    """Create an FFN predictor based on type.

    This factory function provides a unified interface for creating FFN predictors
    that can be used by both Chemprop and CheMeleon models.

    Parameters
    ----------
    ffn_type : str
        FFN architecture type. One of:
        - ``'regression'``: Standard multi-task regression FFN
        - ``'mixture_of_experts'``: MoE with gating network
        - ``'branched'``: Shared trunk with task-specific branches
    input_dim : int
        Input dimension from message passing encoder.
    n_tasks : int
        Number of prediction tasks.
    hidden_dim : int, default=300
        Hidden dimension for FFN layers.
    n_layers : int, default=2
        Number of FFN layers.
    dropout : float, default=0.0
        Dropout probability.
    n_experts : int | None, default=None
        Number of experts for MoE architecture. Defaults to 4 if not specified.
    trunk_n_layers : int | None, default=None
        Number of trunk layers for branched architecture. Defaults to 2 if not specified.
    trunk_hidden_dim : int | None, default=None
        Hidden dimension for trunk in branched architecture. Defaults to hidden_dim.
    task_groups : list[list[int]] | None, default=None
        Task groupings for branched architecture. Defaults to one task per branch.
    criterion : ChempropMetric | None, default=None
        Loss criterion for training.
    task_weights : Tensor | None, default=None
        Per-task loss weights.
    output_transform : UnscaleTransform | None, default=None
        Output transform to apply after forward pass.

    Returns
    -------
    nn.Predictor
        Configured FFN predictor instance.

    Raises
    ------
    ValueError
        If ``ffn_type`` is not one of the supported types.

    Examples
    --------
    Create a standard regression FFN:

    >>> ffn = create_ffn_predictor(
    ...     ffn_type="regression",
    ...     input_dim=300,
    ...     n_tasks=5,
    ... )

    Create a Mixture of Experts FFN:

    >>> ffn = create_ffn_predictor(
    ...     ffn_type="mixture_of_experts",
    ...     input_dim=300,
    ...     n_tasks=5,
    ...     n_experts=4,
    ... )

    Create a Branched FFN with custom task groups:

    >>> ffn = create_ffn_predictor(
    ...     ffn_type="branched",
    ...     input_dim=300,
    ...     n_tasks=5,
    ...     task_groups=[[0, 1], [2, 3, 4]],
    ...     trunk_n_layers=2,
    ... )
    """
    if ffn_type == "regression":
        return RegressionFFN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            n_tasks=n_tasks,
            criterion=criterion,
            task_weights=task_weights,
            output_transform=output_transform,
        )

    elif ffn_type == "mixture_of_experts":
        return MixtureOfExpertsRegressionFFN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            n_tasks=n_tasks,
            n_experts=n_experts or 4,
            criterion=criterion,
            task_weights=task_weights,
            output_transform=output_transform,
        )

    elif ffn_type == "branched":
        if task_groups is None:
            task_groups = [[i] for i in range(n_tasks)]

        return BranchedFFN(
            task_groups=task_groups,
            n_tasks=n_tasks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            trunk_n_layers=trunk_n_layers or 2,
            trunk_hidden_dim=trunk_hidden_dim or hidden_dim,
            criterion=criterion,
            task_weights=task_weights,
            output_transform=output_transform,
        )

    else:
        supported = ["regression", "mixture_of_experts", "branched"]
        raise ValueError(f"Unknown ffn_type: {ffn_type!r}. Supported types: {supported}")
