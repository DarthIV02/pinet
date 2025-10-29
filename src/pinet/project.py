"""Implementation of the projection layer (PyTorch version, GPU-ready)."""

import torch
from functools import partial
from typing import Callable, Optional

from .constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
)
from .dataclasses import EquilibrationParams, ProjectionInstance
from .equilibration import ruiz_equilibration
from .solver import build_iteration_step, initialize


class Project:
    """Projection layer implemented via Douglas-Rachford (PyTorch version)."""

    def __init__(
        self,
        eq_constraint: EqualityConstraint = None,
        ineq_constraint: AffineInequalityConstraint = None,
        box_constraint: BoxConstraint = None,
        unroll: bool = False,
        equilibration_params: EquilibrationParams = EquilibrationParams(),
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        """Initialize the projection layer."""
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.unroll = unroll
        self.equilibration_params = equilibration_params
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.setup()

    def setup(self) -> None:
        """Setup the projection layer and build the projection operator."""
        constraints = [
            c
            for c in (self.eq_constraint, self.box_constraint, self.ineq_constraint)
            if c
        ]
        assert len(constraints) > 0, "At least one constraint must be provided."
        self.dim = constraints[0].dim

        is_single_simple_constraint = (
            self.ineq_constraint is None and len(constraints) == 1
        )

        self.dim_lifted = self.dim
        self.step_iteration = lambda s_prev, yraw, sigma, omega: s_prev
        self.step_final = self._project_single
        self.single_constraint = constraints[0]

        # Initialize scaling factors on GPU
        self.d_r = torch.ones((1, self.single_constraint.n_constraints, 1), device=self.device)
        self.d_c = torch.ones((1, self.single_constraint.dim, 1), device=self.device)

        if not is_single_simple_constraint:
            # Constraints need to be parsed
            if self.ineq_constraint is not None:
                self.dim_lifted += self.ineq_constraint.n_constraints

            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
            )
            self.lifted_eq_constraint, self.lifted_box_constraint, self.lift = parser.parse(method=None)

            # Perform Ruiz equilibration if possible
            if (
                not self.lifted_eq_constraint.var_A
                and self.lifted_eq_constraint.A.shape[0] == 1
            ):
                scaled_A, self.d_r, self.d_c = ruiz_equilibration(
                    self.lifted_eq_constraint.A[0].to(self.device), self.equilibration_params
                )
                self.lifted_eq_constraint.A = scaled_A.reshape(
                    1,
                    *self.lifted_eq_constraint.A.shape[1:],
                )
                self.d_r = self.d_r.reshape(1, -1, 1)
                self.d_c = self.d_c.reshape(1, -1, 1)
            else:
                n_ineq = (
                    self.ineq_constraint.n_constraints
                    if self.ineq_constraint is not None
                    else 0
                )
                self.d_r = torch.ones((1, self.eq_constraint.n_constraints + n_ineq, 1), device=self.device)
                self.d_c = torch.ones((1, self.dim_lifted, 1), device=self.device)

            self.lifted_eq_constraint.method = "pinv"
            self.lifted_eq_constraint.setup()

            # Scale RHS and box constraints
            self.lifted_eq_constraint.b *= self.d_r
            mask = self.lifted_box_constraint.mask
            scale = self.d_c[:, mask, :]
            self.lifted_box_constraint.scale = 1 / scale
            self.lifted_box_constraint.ub *= self.lifted_box_constraint.scale
            self.lifted_box_constraint.lb *= self.lifted_box_constraint.scale

            # Build ADMM iteration and final step
            self.step_iteration, self.step_final = build_iteration_step(
                self.lifted_eq_constraint,
                self.lifted_box_constraint,
                self.dim,
                self.d_c[:, : self.dim, :],
            )

        project_fn = (
            _project_general
            if (self.unroll or is_single_simple_constraint)
            else _project_general
        )

        self._project = partial(
            project_fn,
            initialize_fn=self.initialize,
            step_iteration=self.step_iteration,
            step_final=self.step_final,
            dim_lifted=self.dim_lifted,
            d_r=self.d_r,
            d_c=self.d_c,
        )

        # Call method shortcut
        self.call = self._project

    def initialize(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Zero initial value for the governing sequence."""
        return initialize(
            yraw=yraw,
            ineq_constraint=self.ineq_constraint,
            box_constraint=self.box_constraint,
            dim=self.dim,
            dim_lifted=self.dim_lifted,
            d_r=self.d_r,
        )

    def cv(self, y: ProjectionInstance) -> torch.Tensor:
        """Compute the constraint violation."""
        if y.x.shape[1] != self.dim_lifted:
            y = self.lift(y)
        return torch.maximum(
            self.lifted_eq_constraint.cv(y),
            self.lifted_box_constraint.cv(y),
        )

    def _project_single(self, yraw: ProjectionInstance) -> torch.Tensor:
        """Project a batch of points with a single constraint."""
        if yraw.eq and yraw.eq.A is not None:
            Apinv = torch.linalg.pinv(yraw.eq.A)
            yraw = yraw.update(eq=yraw.eq.update(Apinv=Apinv))
        return self.single_constraint.project(yraw)


# -------------------------------------------------------------------
# General projection: replaces JAX scan with explicit PyTorch loop
# -------------------------------------------------------------------
def _project_general(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, torch.Tensor, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], torch.Tensor],
    dim_lifted: int,
    d_r: torch.Tensor,
    d_c: torch.Tensor,
    yraw: ProjectionInstance,
    s0: Optional[ProjectionInstance] = None,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    **kwargs
) -> tuple[ProjectionInstance, ProjectionInstance]:
    """Douglas-Rachford projection implemented in PyTorch."""
    device = yraw.x.device

    if n_iter > 0:
        s = initialize_fn(yraw) if s0 is None else s0
        for _ in range(n_iter):
            s = step_iteration(s, yraw, sigma, omega)
        sk = s
    else:
        sk = yraw

    y = step_final(sk).x[:, : yraw.x.shape[1], :]
    y_scaled = y * d_c[:, : yraw.x.shape[1], :].to(device)

    return yraw.update(x=y_scaled), sk
