"""Projection layers using other approaches (PyTorch version)."""

from typing import Callable
import torch
import cvxpy as cp
import numpy as np


def get_torch_projection(
    A: torch.Tensor, C: torch.Tensor, d: torch.Tensor, dim: int, tol: float = 1e-3
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compute a batched projection function for polyhedral constraints using CVXPY solver.
    This replaces jaxopt.OSQP.

    Solve for each input xx and equality rhs bb:
        minimize   0.5 ||x - xx||^2
        subject to A x = bb
                   C x <= d

    Args:
        A (torch.Tensor): Equality constraint matrix (shape: [n_eq, dim]).
        C (torch.Tensor): Inequality constraint matrix (shape: [n_ineq, dim]).
        d (torch.Tensor): Upper bounds for inequalities (shape: [n_ineq]).
        dim (int): Dimension of the variable x.
        tol (float): Solver tolerance (not used directly in cvxpy).

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: Projection function.
    """
    A_np = A.cpu().numpy()
    C_np = C.cpu().numpy()
    d_np = d.cpu().numpy()
    n_eq = A.shape[0]

    def project(xx: torch.Tensor, bb: torch.Tensor) -> torch.Tensor:
        xx_np = xx.detach().cpu().numpy()
        bb_np = bb.detach().cpu().numpy()
        proj_batch = []

        for xvec, bvec in zip(xx_np, bb_np):
            y = cp.Variable(dim)
            constraints = [A_np @ y == bvec, C_np @ y <= d_np]
            objective = cp.Minimize(0.5 * cp.sum_squares(y - xvec))
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, eps_abs=tol, eps_rel=tol, warm_start=True, verbose=False)
            proj_batch.append(y.value)

        proj_batch = torch.tensor(np.stack(proj_batch), dtype=xx.dtype, device=xx.device)
        return proj_batch

    return project


def get_cvxpy_projection(
    A: torch.Tensor,
    C: torch.Tensor,
    d: torch.Tensor,
    dim: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Construct a CVXPY-based projection layer.

    Solve for each input xx and equality rhs bb:
        minimize ||y - xx||^2
        subject to A y = bb
                   C y <= d

    Args:
        A (torch.Tensor): Equality constraint matrix [n_eq, dim].
        C (torch.Tensor): Inequality constraint matrix [n_ineq, dim].
        d (torch.Tensor): Upper bounds for inequalities [n_ineq].
        dim (int): Dimension of the variable x.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: Projection function.
    """
    A_np = A.cpu().numpy()
    C_np = C.cpu().numpy()
    d_np = d.cpu().numpy()
    n_eq = A.shape[0]

    def project(xx: torch.Tensor, bb: torch.Tensor) -> torch.Tensor:
        xx_np = xx.detach().cpu().numpy()
        bb_np = bb.detach().cpu().numpy()
        proj_batch = []

        for xvec, bvec in zip(xx_np, bb_np):
            y = cp.Variable(dim)
            constraints = [A_np @ y == bvec, C_np @ y <= d_np]
            objective = cp.Minimize(cp.sum_squares(y - xvec))
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            proj_batch.append(y.value)

        proj_batch = torch.tensor(np.stack(proj_batch), dtype=xx.dtype, device=xx.device)
        return proj_batch

    return project
