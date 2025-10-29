"""Affine inequality constraint module (PyTorch version)."""

from typing import Optional
import torch

from pinet.dataclasses import ProjectionInstance
from .base import Constraint


class AffineInequalityConstraint(Constraint):
    """Affine inequality constraint set.

    The (affine) inequality constraint set is defined as:
        lb <= C @ x <= ub
    where the matrix C and the vectors lb and ub are the parameters.
    """

    def __init__(
        self,
        C: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the affine inequality constraint.

        Args:
            C (torch.Tensor): The matrix C in the inequality.
                Shape (batch_size, n_constraints, dimension).
            lb (torch.Tensor): The lower bound in the inequality.
                Shape (batch_size, n_constraints, 1).
            ub (torch.Tensor): The upper bound in the inequality.
                Shape (batch_size, n_constraints, 1).
        """
        # Auto-detect device if not provided
        if device is None:
            device = C.device

        # Move everything to the same device and dtype
        self.C = C.to(device=device, dtype=dtype)
        self.lb = lb.to(device=device, dtype=dtype)
        self.ub = ub.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        # --- Validity checks ---
        assert (
            self.C.shape[0] == self.lb.shape[0]
            or self.C.shape[0] == 1
            or self.lb.shape[0] == 1
        ), f"Batch sizes inconsistent: C{self.C.shape}, lb{self.lb.shape}"

        assert (
            self.C.shape[0] == self.ub.shape[0]
            or self.C.shape[0] == 1
            or self.ub.shape[0] == 1
        ), f"Batch sizes inconsistent: C{self.C.shape}, ub{self.ub.shape}"

        assert self.C.shape[1] == self.lb.shape[1], (
            "Number of rows in C must equal number of rows in lb."
        )
        assert self.C.shape[1] == self.ub.shape[1], (
            "Number of rows in C must equal number of rows in ub."
        )

    def project(self, inp: ProjectionInstance) -> ProjectionInstance:
        """Project x onto the affine inequality constraint set.

        Args:
            inp (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
                Shape (batch_size, dimension, 1).
        """
        raise NotImplementedError(
            "The 'project' method is not implemented and should not be called."
        )

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.C.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.C.shape[1]

    def cv(self, inp: ProjectionInstance) -> torch.Tensor:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            torch.Tensor: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        Cx = self.C @ inp.x  # Shape: (batch_size, n_constraints, 1)
        cv_ub = torch.clamp(Cx - self.ub, min=0.0)
        cv_lb = torch.clamp(self.lb - Cx, min=0.0)
        cv = torch.maximum(cv_ub, cv_lb)  # Elementwise maximum
        return torch.amax(cv, dim=1, keepdim=True)  # Max violation per batch