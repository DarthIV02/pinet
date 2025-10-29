"""Module for the Alternating Direction Method of Multipliers (ADMM) solver (PyTorch version)."""

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
    """Initialize the ADMM solver state (GPU-ready, PyTorch version).

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

    # Preprocess equality constraints
    if yraw.eq is not None:
        if yraw.eq.A is not None:
            # Lift the equality constraint
            parser = ConstraintParser(
                eq_constraint=EqualityConstraint(yraw.eq.A, yraw.eq.b, method="pinv"),
                ineq_constraint=ineq_constraint,
                box_constraint=box_constraint,
            )
            lifted_eq_constraint, _, _ = parser.parse(method="pinv")

            yraw = yraw.update(
                eq=yraw.eq.update(
                    A=lifted_eq_constraint.A,
                    Apinv=lifted_eq_constraint.Apinv,
                )
            )

        if yraw.eq.b is not None:
            b_lifted = (
                torch.cat(
                    [
                        yraw.eq.b,
                        torch.zeros(
                            (yraw.eq.b.shape[0], dim_lifted - dim, 1), device=device
                        ),
                    ],
                    dim=1,
                )
                * d_r
            )
            yraw = yraw.update(eq=yraw.eq.update(b=b_lifted))

    # Initialize x in the lifted dimension
    return yraw.update(x=torch.zeros((yraw.x.shape[0], dim_lifted, 1), device=device))


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: torch.Tensor = torch.tensor(1.0),
) -> Tuple[
    Callable[[ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance],
    Callable[[ProjectionInstance], ProjectionInstance],
]:
    """Build the iteration and result retrieval step for the ADMM solver (GPU-ready).

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
    device = eq_constraint.A.device if hasattr(eq_constraint, "A") else torch.device("cpu")
    scale = scale.to(device)

    def iteration_step(
        sk: ProjectionInstance,
        yraw: ProjectionInstance,
        sigma: float = 1.0,
        omega: float = 1.7,
    ) -> ProjectionInstance:
        """One iteration of the ADMM solver.

        Args:
            sk (ProjectionInstance): Current ADMM state.
                .x shape (batch_size, lifted_dimension, 1)
            yraw (ProjectionInstance): Point to be projected.
                .x shape (batch_size, dimension, 1)
            sigma (float): ADMM parameter.
            omega (float): ADMM parameter.

        Returns:
            ProjectionInstance: Next ADMM state iterate.
        """
        device = sk.x.device

        # 1. Equality projection
        zk = eq_constraint.project(sk)

        # 2. Reflection step
        reflect = 2 * zk.x - sk.x

        # 3. Compute input for box projection
        tobox = torch.cat(
            (
                (2 * sigma * scale * yraw.x + reflect[:, :dim, :])
                / (1 + 2 * sigma * scale**2),
                reflect[:, dim:, :],
            ),
            dim=1,
        )

        # 4. Box projection
        tk = box_constraint.project(sk.update(x=tobox))

        # 5. ADMM update
        sk = sk.update(x=sk.x + omega * (tk.x - zk.x))

        return sk

    # Second function: extract the projected result
    return iteration_step, lambda y: eq_constraint.project(y)