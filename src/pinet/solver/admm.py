"""Module for the Alternating Direction Method of Multipliers (ADMM) solver (PyTorch version, double precision)."""

from typing import Callable, Tuple
import torch

from pinet.constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
)
from pinet.dataclasses import ProjectionInstance


def initialize(
    yraw: ProjectionInstance,
    ineq_constraint: AffineInequalityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    dim_lifted: int,
    d_r: torch.Tensor,
) -> ProjectionInstance:
    """Initialize the ADMM solver state (GPU-ready, PyTorch version, double precision).

    Args:
        yraw (ProjectionInstance): Point to be projected. Shape (batch_size, dimension, 1)
        ineq_constraint (AffineInequalityConstraint): Inequality constraint.
        box_constraint (BoxConstraint): Box constraint.
        dim (int): Dimension of the original problem.
        dim_lifted (int): Dimension of the lifted problem.
        d_r (torch.Tensor): Scaling factor for the lifted dimension.

    Returns:
        ProjectionInstance: Initial state for the ADMM solver.
    """
    device = yraw.x.device
    dtype = torch.float64

    # Ensure d_r is double precision
    d_r = d_r.to(device=device, dtype=dtype)

    # Preprocess equality constraints
    if yraw.eq is not None:
        if yraw.eq.A is not None:
            # Lift the equality constraint
            parser = ConstraintParser(
                eq_constraint=EqualityConstraint(
                    yraw.eq.A.to(dtype=dtype),
                    yraw.eq.b.to(dtype=dtype) if yraw.eq.b is not None else None,
                    method="pinv",
                ),
                ineq_constraint=ineq_constraint,
                box_constraint=box_constraint,
            )
            lifted_eq_constraint, _, _ = parser.parse(method="pinv")

            yraw = yraw.update(
                eq=yraw.eq.update(
                    A=lifted_eq_constraint.A.to(dtype=dtype),
                    Apinv=lifted_eq_constraint.Apinv.to(dtype=dtype),
                )
            )

        if yraw.eq.b is not None:
            b_lifted = (
                torch.cat(
                    [
                        yraw.eq.b.to(dtype=dtype),
                        torch.zeros(
                            (yraw.eq.b.shape[0], dim_lifted - dim, 1),
                            device=device,
                            dtype=dtype,
                        ),
                    ],
                    dim=1,
                )
                * d_r
            )
            yraw = yraw.update(eq=yraw.eq.update(b=b_lifted))

    # Initialize x in the lifted dimension with double precision
    return yraw.update(
        x=torch.zeros(
            (yraw.x.shape[0], dim_lifted, 1), device=device, dtype=dtype
        )
    )


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: torch.Tensor = torch.tensor(1.0, dtype=torch.float64),
) -> Tuple[
    Callable[[ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance],
    Callable[[ProjectionInstance], ProjectionInstance],
]:
    """Build the iteration and result retrieval step for the ADMM solver (GPU-ready, double precision).

    Args:
        eq_constraint (EqualityConstraint): (Lifted) Equality constraint.
        box_constraint (BoxConstraint): (Lifted) Box constraint.
        dim (int): Dimension of the original problem.
        scale (torch.Tensor): Scaling of primal variables.

    Returns:
        tuple[
            Callable[[ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance],
            Callable[[ProjectionInstance], ProjectionInstance]
        ]:
            The first element is the iteration step,
            the second element is the result retrieval step.
    """
    device = (
        eq_constraint.A.device if hasattr(eq_constraint, "A") else torch.device("cpu")
    )
    dtype = torch.float64
    scale = scale.to(device=device, dtype=dtype)

    def iteration_step(
        sk: ProjectionInstance,
        yraw: ProjectionInstance,
        sigma: float = 1.0,
        omega: float = 1.7,
    ) -> ProjectionInstance:
        """One iteration of the ADMM solver (double precision)."""
        device = sk.x.device

        # 1. Equality projection
        zk = eq_constraint.project(sk)

        # 2. Reflection step
        reflect = 2.0 * zk.x - sk.x

        # 3. Compute input for box projection
        sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        omega_t = torch.tensor(omega, device=device, dtype=dtype)

        tobox = torch.cat(
            (
                (2.0 * sigma_t * scale * yraw.x + reflect[:, :dim, :])
                / (1.0 + 2.0 * sigma_t * scale**2),
                reflect[:, dim:, :],
            ),
            dim=1,
        )

        # 4. Box projection
        tk = box_constraint.project(sk.update(x=tobox))

        # 5. ADMM update
        sk = sk.update(x=sk.x + omega_t * (tk.x - zk.x))

        return sk

    # Second function: extract the projected result
    return iteration_step, lambda y: eq_constraint.project(y)