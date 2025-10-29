"""Module for setting HCNN models for the benchmarks (PyTorch version)."""

import time
from typing import Callable, Optional

import torch
import torch.nn as nn

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
)

from .other_projections import get_cvxpy_projection, get_torch_projection  # changed get_jaxopt_projection -> torch version


def setup_pinet(
    hyperparameters: dict,
    eq_constraint: Optional[EqualityConstraint] = None,
    ineq_constraint: Optional[AffineInequalityConstraint] = None,
    box_constraint: Optional[BoxConstraint] = None,
    setup_reps: int = -1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Setup of pinet projection layer."""
    projection_layer = Project(
        ineq_constraint=ineq_constraint,
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        unroll=hyperparameters["unroll"],
        equilibration_params=EquilibrationParams(**hyperparameters["equilibrate"]),
        device=device,
    )

    setup_time = 0.0
    if setup_reps > 0:
        start_setup_time = time.time()
        for _ in range(setup_reps):
            _ = Project(
                ineq_constraint=ineq_constraint,
                eq_constraint=eq_constraint,
                box_constraint=box_constraint,
                unroll=hyperparameters["unroll"],
                equilibration_params=EquilibrationParams(**hyperparameters["equilibrate"]),
                device=device,
            )
        setup_time = (time.time() - start_setup_time) / max(setup_reps, 1)
        print(f"Time to create constraints: {setup_time:.5f} seconds")

    kw = {} if hyperparameters["unroll"] else {"n_iter_bwd": hyperparameters["n_iter_bwd"], "fpi": hyperparameters["fpi"]}

    def project(x: torch.Tensor, b: torch.Tensor):
        inp = ProjectionInstance(x=x[..., None], eq=EqualityConstraintsSpecification(b=b))
        return projection_layer.call(
            yraw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_train"],
            **kw,
        )[0].x[..., 0]

    def project_test(x: torch.Tensor, b: torch.Tensor):
        inp = ProjectionInstance(x=x[..., None], eq=EqualityConstraintsSpecification(b=b))
        return projection_layer.call(
            yraw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_test"],
            **kw,
        )[0].x[..., 0]

    return project, project_test, setup_time


def setup_cvxpy(
    A: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    ub: torch.Tensor,
    setup_reps: int,
    hyperparameters: dict,
):
    """Setup cvxpy projection layer in PyTorch."""
    cvxpy_proj = get_cvxpy_projection(A=A[0, :, :].cpu().numpy(), C=C[0, :, :].cpu().numpy(), d=ub[0, :, 0].cpu().numpy(), dim=A.shape[2])

    def project(xx: torch.Tensor, bb: torch.Tensor):
        return torch.tensor(
            cvxpy_proj(xx.cpu().numpy(), bb[:, :, 0].cpu().numpy(), solver_args={
                "verbose": False,
                "eps_abs": hyperparameters["cvxpy_tol"],
                "eps_rel": hyperparameters["cvxpy_tol"],
            })[0],
            dtype=torch.float32,
            device=xx.device,
        )

    project_test = project
    return project, project_test, 0.0


class HardConstrainedMLP(nn.Module):
    """MLP that mimics JAX: features_list defines hidden layers, input dim inferred from first batch."""

    def __init__(
        self,
        project: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        project_test: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dim: int,
        features_list: list[int],
        activation: nn.Module = nn.ReLU,
        raw_train: bool = False,
        raw_test: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.features_list = features_list
        self.activation = activation()
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.project = project
        self.project_test = project_test

        self.net = None  # will initialize on first forward pass

    def _build_net(self, input_dim: int, device: torch.device):
        """Build the network given the inferred input dimension."""
        layers = []
        in_dim = input_dim
        for features in self.features_list:
            layers.append(nn.Linear(in_dim, features, device=device))
            layers.append(self.activation)
            in_dim = features
        layers.append(nn.Linear(in_dim, self.dim, device=device))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, b: torch.Tensor, test: bool = False, device: Optional[torch.device] = None):
        if device is None:
            device = x.device  # fallback: use input batch device

        if self.net is None:
            # Infer input dimension from first batch
            self._build_net(x.shape[1], device)

        x = self.net(x.to(device))

        if test and not self.raw_test:
            x = self.project_test(x.to(device), b.to(device))
        elif not test and not self.raw_train:
            x = self.project(x.to(device), b.to(device))
        return x

def build_model_and_train_step(
    dim: int,
    features_list: list,
    activation: nn.Module,
    project,
    project_test,
    raw_train: bool,
    raw_test: bool,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    example_x: torch.Tensor,
    example_b: torch.Tensor,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Build PyTorch model and training step."""
    model = HardConstrainedMLP(
        project=project,
        project_test=project_test,
        dim=dim,
        features_list=features_list,
        activation=activation,
        raw_train=raw_train,
        raw_test=raw_test,
    ).to(device)

    example_x = example_x.to(device)
    example_b = example_b.to(device)

    def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, x_batch: torch.Tensor, b_batch: torch.Tensor):
        model.train()
        x_batch = x_batch.to(device)
        b_batch = b_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch, b_batch, test=False)
        loss = loss_fn(preds, b_batch).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    return model, train_step