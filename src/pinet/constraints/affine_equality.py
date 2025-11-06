"""Equality constraint module (PyTorch version, double precision)."""

from typing import Optional, Tuple
import torch

from pinet.dataclasses import ProjectionInstance  # assuming this is compatible
from .base import Constraint


class EqualityConstraint(Constraint):
    """Equality constraint set (double precision).

    The (affine) equality constraint set is defined as:
        A @ x == b
    where the matrix A and the vector b are parameters.
    """

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        method: Optional[str] = "pinv",
        var_b: Optional[bool] = False,
        var_A: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the equality constraint (double precision)."""
        assert A is not None, "Matrix A must be provided."
        assert b is not None, "Vector b must be provided."

        if device is None:
            device = A.device

        dtype = torch.float64
        self.A = A.to(device=device, dtype=dtype)
        self.b = b.to(device=device, dtype=dtype)
        self.method = method
        self.var_b = var_b
        self.var_A = var_A
        self.setup()

    def setup(self) -> None:
        """Set up the equality constraint."""
        assert self.A.ndim == 3, (
            "A must have shape (batch_size, n_constraints, dimension)."
        )
        assert self.b.ndim == 3, (
            "b must have shape (batch_size, n_constraints, 1)."
        )
        assert self.b.shape[2] == 1, (
            "b must have shape (batch_size, n_constraints, 1)."
        )

        # Check batch sizes
        assert (
            self.A.shape[0] == self.b.shape[0]
            or self.A.shape[0] == 1
            or self.b.shape[0] == 1
        ), f"Batch sizes inconsistent: A{self.A.shape}, b{self.b.shape}"

        assert self.A.shape[1] == self.b.shape[1], (
            "Number of rows in A must equal size of b."
        )

        if self.method == "pinv":
            if not self.var_A:
                self.Apinv = torch.linalg.pinv(self.A)
            self.project = self.project_pinv
        else:
            def default_project(yraw):
                raise NotImplementedError("No projection method set.")
            self.project = default_project

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Forward project method required by Constraint ABC."""
        if self.method == "pinv":
            return self.project_pinv(yraw)
        else:
            raise NotImplementedError("No projection method set.")

    def get_params(
        self, inp: ProjectionInstance
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get A, b, Apinv depending on varying constraints."""
        b = inp.eq.b.double() if inp.eq and inp.eq.b is not None else self.b
        A = inp.eq.A.double() if inp.eq and self.var_A else self.A
        Apinv = inp.eq.Apinv.double() if inp.eq and self.var_A else getattr(self, "Apinv", None)
        return b, A, Apinv

    def project_pinv(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto equality constraints using pseudo-inverse."""
        b, A, Apinv = self.get_params(yraw)
        correction = A @ yraw.x.double() - b
        projected_x = yraw.x.double() - Apinv @ correction
        return yraw.update(x=projected_x)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.A.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.A.shape[1]

    def cv(self, inp: ProjectionInstance) -> torch.Tensor:
        """Compute the constraint violation (double precision)."""
        b, A, _ = self.get_params(inp)
        violation = torch.linalg.norm(A @ inp.x.double() - b, ord=float("inf"), dim=1, keepdim=True)
        return violation.unsqueeze(-1)