from typing import List, Sequence

import torch
import torch.nn.functional as F
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.nn import Predictor, PredictorRegistry
from chemprop.nn.metrics import MSE, ChempropMetric
from chemprop.nn.predictors import MLP
from chemprop.nn.transforms import UnscaleTransform
from chemprop.utils import Factory
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor, nn


@PredictorRegistry.register("regression-moe")
class MixtureOfExpertsRegressionFFN(Predictor, HyperparametersMixin):
    r"""
    Implementation of the Adaptive Mixture of Local Experts [1]_ model for regression tasks.
    The works by passing the learned representation from message passing into one "gating network"
    and a configurable number of "experts". The outputs of the individual experts are
    multiplied element-wise by the output of the gating network, enabling the overall
    architecture to 'specialize' experts in certain types of inputs dynamically during
    training.

    References
    ----------
    .. [1] R. A. Jacobs, M. I. Jordan, S. J. Nowlan and G. E. Hinton, "Adaptive Mixtures of Local Experts"
        Neural Computation, vol. 3, no. 1, pp. 79-87, March 1991, doi: 10.1162/neco.1991.3.1.79.
    """

    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_experts: int = 2,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
        gate_hidden_dim: int | None = None,
        gate_n_layers: int = 1,
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        """Adaptive Mixture of Local Experts Regression network

        Args:
            n_experts (int, optional): Number of expert sub-networks. Defaults to 2.
            n_tasks (int, optional): Output dimension for each sub-network. Defaults to 1.
            input_dim (int, optional): Size of message passing output. Defaults to DEFAULT_HIDDEN_DIM.
            hidden_dim (int, optional): Number of neurons per layer in sub-networks. Defaults to 300.
            n_layers (int, optional): Number of layers per network in sub-networks. Defaults to 1.
            dropout (float, optional): Dropout rate in all networks. Defaults to 0.0.
            activation (str | nn.Module, optional): Choice of activation function for all network. Defaults to "relu".
            gate_hidden_dim (int | None, optional): Number of neurons in gating network. Defaults to None.
            gate_n_layers (int, optional): Number of layers in gating network. Defaults to 1.
            criterion (ChempropMetric | None, optional): Criterion for training. Defaults to None.
            task_weights (Tensor | None, optional): Weights for each individual task. Defaults to None.
            threshold (float | None, optional): Passed to criterion. Defaults to None.
            output_transform (UnscaleTransform | None, optional): Output transform
                to be applied after forward. Defaults to None.
        """
        super().__init__()
        ignore_list = ["criterion", "output_transform", "activation"]
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__

        self.n_experts = n_experts

        # Experts
        self.experts = nn.ModuleList(
            [
                MLP.build(
                    input_dim,
                    n_tasks * self.n_targets,
                    hidden_dim,
                    n_layers,
                    dropout,
                    activation,
                )
                for _ in range(n_experts)
            ]
        )

        # Gating network
        gate_hidden_dim = gate_hidden_dim or hidden_dim
        self.gate = MLP.build(
            input_dim,
            n_experts,
            gate_hidden_dim,
            gate_n_layers,
            dropout,
            activation,
        )

        # Criterion
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        self.criterion = criterion or Factory.build(
            self._T_default_criterion, task_weights=task_weights, threshold=threshold
        )

        self.output_transform = output_transform if output_transform is not None else nn.Identity()

    @property
    def input_dim(self) -> int:
        return self.experts[0].input_dim  # type: ignore[return-value]

    @property
    def output_dim(self) -> int:
        return self.experts[0].output_dim  # type: ignore[return-value]

    @property
    def n_tasks(self) -> int:
        return self.output_dim // self.n_targets

    def forward(self, Z: Tensor) -> Tensor:
        expert_outputs = torch.stack([expert(Z) for expert in self.experts], dim=1)  # [B, E, O]
        gate_logits = self.gate(Z)  # [B, E]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, E]

        Y = torch.einsum("be,bed->bd", gate_weights, expert_outputs)
        return self.output_transform(Y)

    train_step = forward

    def encode(self, Z: Tensor, i: int) -> Tensor:
        return self.experts[0][:i](Z)  # type: ignore[index]


@PredictorRegistry.register("regression-branched")
class BranchedFFN(Predictor, HyperparametersMixin):
    """
    Regression Predictor with a shared trunk (optional) followed by multiple
    independent MLP branches, each responsible for a group of tasks.

    Depth indexing for encode(Z, i):
        - i counts blocks across trunk then branches.
        - If i <= trunk_n_layers: returns trunk[:i](Z)
        - Else: returns concat(branch[:i - trunk_n_layers](trunk(Z)) for each branch)

    Parameters
    ----------
    task_groups : list[list[int]]
        Partition of global task indices [0..n_tasks-1].
    n_tasks : int
        Total number of tasks.
    input_dim : int
        Input (fingerprint) dimensionality.
    hidden_dim : int | list[int]
        Branch hidden dim (shared or per-branch).
    n_layers : int | list[int]
        Branch depth (shared or per-branch).
    dropout : float | list[float]
        Branch dropout (shared or per-branch).
    activation : str | nn.Module | list[...]
        Branch activation (shared or per-branch).
    # NEW trunk controls:
    trunk_n_layers : int
        Number of shared layers before branching. 0 disables the trunk.
    trunk_hidden_dim : int
        Hidden size of trunk blocks; also the input dim to branches when trunk_n_layers>0.
    trunk_dropout : float
        Dropout in trunk blocks.
    trunk_activation : str | nn.Module
        Activation used in trunk blocks.
    """

    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        task_groups: List[List[int]],
        *,
        n_tasks: int,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        # branch params
        hidden_dim: int | Sequence[int] = 300,
        n_layers: int | Sequence[int] = 1,
        dropout: float | Sequence[float] = 0.0,
        activation: str | nn.Module | Sequence[str | nn.Module] = "relu",
        # trunk params
        trunk_n_layers: int = 0,
        trunk_hidden_dim: int = 384,
        trunk_dropout: float = 0.0,
        trunk_activation: str | nn.Module = "relu",
        # loss/transform
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        super().__init__()

        ignore = ["criterion", "output_transform", "activation", "trunk_activation"]
        self.save_hyperparameters(ignore=ignore)
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["activation"] = activation
        self.hparams["trunk_activation"] = trunk_activation
        self.hparams["cls"] = self.__class__

        # ---- validate groups ----
        flat = [t for g in task_groups for t in g]
        if sorted(flat) != list(range(n_tasks)):
            raise ValueError(
                "task_groups must be a partition of range(n_tasks) with no gaps/dupes. "
                f"Got n_tasks={n_tasks}, groups={task_groups}"
            )
        self.task_groups = [list(g) for g in task_groups]
        self._n_tasks = n_tasks
        self._input_dim = input_dim

        # ---- trunk (shared pre-branch) ----
        self.trunk_layers = trunk_n_layers
        if trunk_n_layers > 0:
            self.trunk = MLP.build(
                input_dim=input_dim,
                output_dim=trunk_hidden_dim,
                hidden_dim=trunk_hidden_dim,
                n_layers=trunk_n_layers,
                dropout=trunk_dropout,
                activation=trunk_activation,
            )
            branch_input_dim = trunk_hidden_dim
        else:
            self.trunk = nn.Identity()
            branch_input_dim = input_dim

        # ---- normalize per-branch hyperparams ----
        B = len(self.task_groups)

        def _to_list(val, name):
            if isinstance(val, (list, tuple)):
                if len(val) != B:
                    raise ValueError(f"{name} length must match number of branches ({B}).")
                return list(val)
            return [val] * B

        hidden_dim_l = _to_list(hidden_dim, "hidden_dim")
        n_layers_l = _to_list(n_layers, "n_layers")
        dropout_l = _to_list(dropout, "dropout")
        activation_l = _to_list(activation, "activation")

        # ---- build branches ----
        self.branches = nn.ModuleList()
        for g, hdim, nl, do, act in zip(self.task_groups, hidden_dim_l, n_layers_l, dropout_l, activation_l):
            out_dim = len(g) * self.n_targets
            self.branches.append(MLP.build(branch_input_dim, out_dim, hdim, nl, do, act))

        # ---- loss/metric/transform ----
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        self.criterion = criterion or Factory.build(self._T_default_criterion, task_weights=task_weights)
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

        # ---- concat->global reindex ----
        concat_order = [t for g in self.task_groups for t in g]
        inv_perm = torch.empty(self._n_tasks, dtype=torch.long)
        for pos, t in enumerate(concat_order):
            inv_perm[t] = pos
        self.register_buffer("_concat_to_global_idx", inv_perm, persistent=False)

    # -------- properties --------
    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.n_tasks * self.n_targets

    @property
    def n_tasks(self) -> int:
        return self._n_tasks

    # -------- forward/train --------
    def _forward_trunk(self, Z: Tensor) -> Tensor:
        return self.trunk(Z) if self.trunk_layers > 0 else Z

    def _concat_branch_outputs(self, Zb: Tensor) -> Tensor:
        outs = [branch(Zb) for branch in self.branches]
        return torch.cat(outs, dim=1)

    def forward(self, Z: Tensor) -> Tensor:
        Zb = self._forward_trunk(Z)
        Y_concat = self._concat_branch_outputs(Zb)
        Y = Y_concat.index_select(dim=1, index=self._concat_to_global_idx)
        return self.output_transform(Y)

    train_step = forward

    # -------- encoding --------
    def encode(self, Z: Tensor, i: int) -> Tensor:
        """
        If i <= trunk_n_layers: returns trunk[:i](Z)
        Else: runs full trunk, then concatenates branch[:i - trunk_n_layers](trunk(Z))
        """
        if i <= 0:
            return Z  # zero-layer encoding (no-op) to match Predictor.encode semantics

        if self.trunk_layers > 0 and i <= self.trunk_layers:
            return self.trunk[:i](Z)  # type: ignore[index]

        Zb = self._forward_trunk(Z)
        j = i - self.trunk_layers  # remaining depth to slice from branches
        reps = [branch[:j](Zb) for branch in self.branches]  # type: ignore[index]
        return torch.cat(reps, dim=1)
