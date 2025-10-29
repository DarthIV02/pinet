"""Plotting functionalities for toy MPC (PyTorch version, GPU ready)."""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch

from .load_toy_MPC import ToyMPCDataset


# ============================================================
#   TRAINING CURVES PLOTTING
# ============================================================

def plot_training(
    train_loader: ToyMPCDataset,
    valid_loader: ToyMPCDataset,
    training_losses: list,
    validation_losses: list,
    eqcvs: list,
    ineqcvs: list,
) -> None:
    """Plot training curves using PyTorch tensors."""
    opt_train_loss = []
    for _, obj_batch in train_loader:
        opt_train_loss.append(obj_batch)
    opt_train_loss = torch.cat(opt_train_loss, dim=0).mean().item()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.plot(training_losses, label="Training Loss")
    plt.axhline(y=opt_train_loss, color="r", linestyle="-", linewidth=2, label="Optimal Training Objective")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    opt_valid_loss = []
    for _, obj_batch in valid_loader:
        opt_valid_loss.append(obj_batch)
    opt_valid_loss = torch.cat(opt_valid_loss, dim=0).mean().item()

    plt.subplot(1, 4, 2)
    plt.plot(validation_losses, label="Validation Loss")
    plt.axhline(y=opt_valid_loss, color="r", linestyle="-", linewidth=2, label="Optimal Validation Objective")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.semilogy(eqcvs, label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.semilogy(ineqcvs, label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
#   TRAJECTORY GENERATION
# ============================================================

def generate_trajectories(
    model: torch.nn.Module,
    As: torch.Tensor,
    lbxs: torch.Tensor,
    ubxs: torch.Tensor,
    lbus: torch.Tensor,
    ubus: torch.Tensor,
    alpha: float,
    base_dim: int,
    Y_DIM: int,
    dimx: int,
    xhat: torch.Tensor,
    T: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate trajectories from the model and CVXPY solver."""
    ntraj = 1
    xinit = torch.tensor([[-7.0, -5.0]], device=device).reshape(ntraj, base_dim, 1)
    
    # Evaluate the network
    Xinitfull = torch.cat((xinit, torch.zeros((ntraj, As.shape[1] - base_dim, 1), device=device)), dim=1)
    model.eval()
    with torch.no_grad():
        trajectories = model(xinit[:, :, 0].to(device), Xinitfull.to(device), test=True)

    # Solve exact problems with cvxpy
    trajectories_cp = torch.zeros((ntraj, Y_DIM, 1))
    for i in range(ntraj):
        xcp = cp.Variable(Y_DIM)
        xinitcp = cp.Parameter(base_dim)
        constraints = [
            As[0].cpu().numpy() @ xcp == np.hstack([xinitcp.value if xinitcp.value is not None else np.zeros(base_dim), np.zeros(dimx - base_dim)]),
            xcp[:dimx] >= lbxs[0, :, 0].cpu().numpy(),
            xcp[:dimx] <= ubxs[0, :, 0].cpu().numpy(),
            xcp[dimx:] >= lbus[0, :, 0].cpu().numpy(),
            xcp[dimx:] <= ubus[0, :, 0].cpu().numpy(),
        ]
        objective = cp.Minimize(
            cp.sum_squares(xcp[:dimx] - np.tile(xhat[:, 0].cpu().numpy(), T + 1))
            + alpha * cp.sum_squares(xcp[dimx:])
        )
        problem = cp.Problem(objective, constraints)
        xinitcp.value = xinit[i, :, 0].cpu().numpy()
        problem.solve(verbose=False)
        trajectories_cp[i] = torch.tensor(xcp.value, dtype=torch.float32).reshape(-1, 1)

    # Plot trajectories
    for ii in range(ntraj):
        xpred = trajectories[ii, :dimx].cpu().numpy().reshape(T + 1, base_dim)
        xgt = trajectories_cp[ii, :dimx].cpu().numpy().reshape(T + 1, base_dim)
        plt.plot(xpred[:, 0], xpred[:, 1], "-o", label="Prediction")
        plt.plot(xgt[:, 0], xgt[:, 1], "--*", label="Ground Truth")
        plt.plot(xhat[0].cpu(), xhat[1].cpu(), "rx", markersize=10, label="Goal")
        # Bounds rectangle
        rect = plt.Rectangle(
            (lb[0, 0, 0].cpu(), lb[0, 1, 0].cpu()),
            ub[0, 0, 0].cpu() - lb[0, 0, 0].cpu(),
            ub[1, 0, 0].cpu() - lb[1, 0, 0].cpu(),
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
            label="Bounds",
        )
        plt.gca().add_patch(rect)
        plt.legend()
        plt.show()

    return trajectories, trajectories_cp