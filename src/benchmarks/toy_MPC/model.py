"""Module for setting up Pinet models for toy MPC (PyTorch version, GPU ready, double precision)."""

from typing import Any, Callable
import torch
import torch.nn as nn

from pinet import BoxConstraint, BoxConstraintSpecification, EqualityConstraint
from src.benchmarks.model import build_model_and_train_step, setup_pinet


def setup_model(
    hyperparameters: dict[str, Any],
    A: torch.Tensor,
    X: torch.Tensor,
    b: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    batched_objective: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Receives problem (hyper)parameters and returns the model, its parameters, and a train_step.

    All tensors are converted to double precision (torch.float64) for numerical stability.

    Args:
        hyperparameters (dict[str, Any]): Hyperparameters for the model.
        A (torch.Tensor): Coefficient matrix for the equality constraint.
        X (torch.Tensor): Input data for the model.
        b (torch.Tensor): Right-hand side vector for the equality constraint.
        lb (torch.Tensor): Lower bounds for the box constraint.
        ub (torch.Tensor): Upper bounds for the box constraint.
        batched_objective (Callable[[torch.Tensor], torch.Tensor]): Function to compute
            the objective value for the model predictions.
        device (torch.device): Device to place model and tensors on.

    Returns:
        model (nn.Module): The Pinet model.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        train_step (Callable): Function to perform a training step.
    """
    dtype = torch.float64  # enforce double precision

    # Map string names to PyTorch activation classes
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
    }

    # Select activation
    activation_name = hyperparameters["activation"].lower()
    activation = activation_map.get(activation_name)
    if activation is None:
        raise ValueError(f"Unknown activation: {activation_name}")

    # Move tensors to device and convert to double precision
    A = A.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)
    X = X.to(device=device, dtype=dtype)
    lb = lb.to(device=device, dtype=dtype)
    ub = ub.to(device=device, dtype=dtype)

    # Constraints + projection layer
    eq_constraint = EqualityConstraint(A=A, b=b, method=None, var_b=True, device=device)
    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub))
    project, project_test, _ = setup_pinet(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        hyperparameters=hyperparameters,
        device=device,
    )

    example_x = X[:1, :, 0].to(dtype=torch.float64, device=device)
    example_b = b[:1].to(dtype=torch.float64, device=device)

    # Build model and training step
    model, train_step = build_model_and_train_step(
        dim=A.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        project=project,
        project_test=project_test,
        raw_train=hyperparameters.get("raw_train", False),
        raw_test=hyperparameters.get("raw_test", False),
        loss_fn=lambda preds, _b: batched_objective(preds.double()),
        example_x=X[:1, :, 0].double(),
        example_b=b[:1].double(),
        device=device,
    )

    # Force model parameters & buffers to double precision
    model = model.double().to(device)

    # Run one forward pass to build model
    _ = model(X[:1, :, 0], b[:1])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.get("lr", 1e-3))

    return model, optimizer, train_step