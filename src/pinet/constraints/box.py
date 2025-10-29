"""Box constraint module (PyTorch version)."""

import torch
from pinet.dataclasses import BoxConstraintSpecification, ProjectionInstance
from .base import Constraint


class BoxConstraint(Constraint):
    """Box constraint set.

    The box constraint set is defined as the Cartesian product of intervals:
        lb <= x <= ub
    Optionally acts only on a subset of dimensions using a mask.
    """

    def __init__(self, box_spec: BoxConstraintSpecification) -> None:
        """Initialize the box constraint.

        Args:
            box_spec (BoxConstraintSpecification): Specification of the box constraint.
                For variable bounds, provide example bounds.
        """
        self.lb = box_spec.lb
        self.ub = box_spec.ub
        self.mask = box_spec.mask

        # Determine dimension
        self._dim = (
            self.lb.shape[1] if self.lb is not None else self.ub.shape[1]
        )

        # Scale factor placeholder
        self.scale = torch.ones((1, self._dim, 1),
                                device=(self.lb.device if self.lb is not None else "cpu"))

        # Default mask: all dimensions active
        if self.mask is None:
            self.mask = torch.ones(
                self._dim, dtype=torch.bool, device=self.scale.device
            )

    def get_params(
        self, yraw: ProjectionInstance
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the parameters of the box constraint.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to get parameters from.

        Returns:
            (lb, ub, mask): Tensors for lower/upper bounds and active dimension mask.
        """
        lb = (
            (yraw.box.lb * self.scale)
            if yraw.box and yraw.box.lb is not None
            else self.lb
        )
        ub = (
            (yraw.box.ub * self.scale)
            if yraw.box and yraw.box.ub is not None
            else self.ub
        )
        mask = yraw.box.mask if yraw.box and yraw.box.mask is not None else self.mask

        device = yraw.x.device
        lb = lb.to(device) if lb is not None else -torch.inf * torch.ones_like(ub)
        ub = ub.to(device) if ub is not None else torch.inf * torch.ones_like(lb)
        mask = mask.to(device)

        return lb, ub, mask

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project x onto the feasible region defined by the box.

        Args:
            yraw (ProjectionInstance): Input to project. The .x attribute is used.

        Returns:
            ProjectionInstance: The projected instance with updated x.
        """
        lb, ub, mask = self.get_params(yraw)

        # Apply projection only on masked dimensions
        x_proj = yraw.x.clone()
        x_proj[:, mask, :] = torch.clamp(yraw.x[:, mask, :], lb, ub)
        return yraw.update(x=x_proj)

    def cv(self, y: ProjectionInstance) -> torch.Tensor:
        """Compute the constraint violation.

        Args:
            y (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            torch.Tensor: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1)
        """
        lb, ub, mask = self.get_params(y)
        x_masked = y.x[:, mask, :]

        cv_ub = torch.max(x_masked - ub, dim=1, keepdim=True).values
        cv_lb = torch.max(lb - x_masked, dim=1, keepdim=True).values
        cvs = torch.maximum(cv_ub, cv_lb)
        return torch.maximum(cvs, torch.zeros_like(cvs))

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self._dim