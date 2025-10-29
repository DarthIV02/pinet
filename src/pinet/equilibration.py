"""Modified Ruiz equilibration (PyTorch version, GPU-ready)."""

import torch
from .dataclasses import EquilibrationParams


def ruiz_equilibration(
    A: torch.Tensor, params: EquilibrationParams
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform modified Ruiz equilibration on matrix A using PyTorch (GPU-ready).

    Ruiz equilibration iteratively scales rows and columns of A so that
    all rows and all columns have approximately equal norms.

    Args:
        A (torch.Tensor): Input matrix with shape (n_r, n_c).
        params (EquilibrationParams): Parameters for equilibration.

    Returns:
        scaled_A (torch.Tensor): Equilibrated matrix.
        d_r (torch.Tensor): Row scaling factors such that scaled_A = diag(d_r) @ A @ diag(d_c).
        d_c (torch.Tensor): Column scaling factors.
    """
    assert A.ndim == 2, "Input matrix to equilibration must be 2-dimensional."

    device = A.device
    scaled_A = A.clone()

    d_r = torch.ones(A.shape[0], device=device)
    d_c = torch.ones(A.shape[1], device=device)

    # Keep track of best criterion
    best_criterion = torch.tensor(1.0, device=device)
    d_r_best = d_r.clone()
    d_c_best = d_c.clone()

    # Initialize column scaling
    alpha = (
        (A.shape[0] / A.shape[1]) ** (1 / (2 * params.ord))
        if params.col_scaling
        else 1.0
    )

    for _ in range(params.max_iter):
        if params.update_mode == "Gauss":
            # ---- Scale rows ----
            row_norms = torch.linalg.norm(scaled_A, dim=1, ord=params.ord)
            row_factors = torch.where(row_norms > 0, torch.sqrt(row_norms), torch.ones_like(row_norms))
            d_r = d_r / row_factors
            scaled_A = scaled_A / row_factors[:, None]

            # ---- Scale columns ----
            col_norms = torch.linalg.norm(scaled_A, dim=0, ord=params.ord)
            col_factors = alpha * torch.where(col_norms > 0, torch.sqrt(col_norms), torch.ones_like(col_norms))
            d_c = d_c / col_factors
            scaled_A = scaled_A / col_factors[None, :]

        else:  # Jacobi update
            row_norms = torch.linalg.norm(scaled_A, dim=1, ord=params.ord)
            row_factors = torch.where(row_norms > 0, torch.sqrt(row_norms), torch.ones_like(row_norms))

            col_norms = torch.linalg.norm(scaled_A, dim=0, ord=params.ord)
            col_factors = alpha * torch.where(col_norms > 0, torch.sqrt(col_norms), torch.ones_like(col_norms))

            d_r = d_r / row_factors
            d_c = d_c / col_factors
            scaled_A = scaled_A / row_factors[:, None]
            scaled_A = scaled_A / col_factors[None, :]

        # ---- Convergence check ----
        new_row_norms = torch.linalg.norm(scaled_A, dim=1, ord=params.ord)
        new_col_norms = torch.linalg.norm(scaled_A, dim=0, ord=params.ord)

        term_criterion = torch.maximum(
            1 - torch.min(new_row_norms) / torch.max(new_row_norms),
            1 - torch.min(new_col_norms) / torch.max(new_col_norms),
        )

        # Track best scaling so far
        if term_criterion < best_criterion:
            best_criterion = term_criterion
            d_r_best = d_r.clone()
            d_c_best = d_c.clone()

        if term_criterion < params.tol:
            break

    # ---- Apply best scaling ----
    scaled_A_best = A * d_r_best[:, None]
    scaled_A_best = scaled_A_best * d_c_best[None, :]

    # ---- Safeguard (avoid worsening conditioning) ----
    if params.safeguard:
        cond_A = torch.linalg.cond(A)
        cond_scaled_A = torch.linalg.cond(scaled_A_best)
        if cond_scaled_A > cond_A:
            scaled_A_best = A.clone()
            d_r_best = torch.ones(A.shape[0], device=device)
            d_c_best = torch.ones(A.shape[1], device=device)

    return scaled_A_best, d_r_best, d_c_best