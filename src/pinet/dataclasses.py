"""This file contains dataclasses used to encapsulate inputs for the Pinet layer (PyTorch version)."""

from dataclasses import dataclass, replace
from typing import Optional
import torch


@dataclass(frozen=True)
class EqualityConstraintsSpecification:
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b (Optional[torch.Tensor]): Vector representing the RHS of the equality constraint.
            Shape (batch_size, n_constraints, 1)
        A (Optional[torch.Tensor]): Matrix representing the LHS of the equality constraint.
            Shape (batch_size, n_constraints, dimension).
        Apinv (Optional[torch.Tensor]): The pseudoinverse of the matrix A.
            Shape (batch_size, dimension, n_constraints).
    """

    b: Optional[torch.Tensor] = None
    A: Optional[torch.Tensor] = None
    Apinv: Optional[torch.Tensor] = None

    def validate(self) -> None:
        """Validate the equality constraints specification."""
        if self.A is not None and self.b is None:
            raise ValueError("If A is provided, b must also be provided.")

    def update(self, **kwargs) -> "EqualityConstraintsSpecification":
        """Return a new instance with updated attributes."""
        return replace(self, **kwargs)

    def to(self, device: torch.device) -> "EqualityConstraintsSpecification":
        """Move tensors to a given device (e.g., GPU)."""
        return EqualityConstraintsSpecification(
            b=self.b.to(device) if self.b is not None else None,
            A=self.A.to(device) if self.A is not None else None,
            Apinv=self.Apinv.to(device) if self.Apinv is not None else None,
        )


@dataclass(frozen=True)
class BoxConstraintSpecification:
    """Dataclass representing inputs used in forming box constraints.

    Attributes:
        lb (torch.Tensor): Lower bound of the box. Shape (batch_size, n_constraints, 1).
        ub (torch.Tensor): Upper bound of the box. Shape (batch_size, n_constraints, 1).
        mask (Optional[torch.Tensor]): Mask to apply the constraint only to some dimensions.
    """

    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

    def update(self, **kwargs) -> "BoxConstraintSpecification":
        """Return a new instance with updated attributes."""
        return replace(self, **kwargs)

    def to(self, device: torch.device) -> "BoxConstraintSpecification":
        """Move tensors to a given device (e.g., GPU)."""
        return BoxConstraintSpecification(
            lb=self.lb.to(device) if self.lb is not None else None,
            ub=self.ub.to(device) if self.ub is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
        )

    def validate(self) -> None:
        """Validate the box constraint specification."""
        if self.lb is None and self.ub is None:
            raise ValueError("At least one of lower or upper bounds must be provided.")

        if self.lb is not None and self.lb.ndim != 3:
            raise ValueError(
                f"Lower bound must have shape (batch_size, n_constraints, 1). Got {self.lb.shape}."
            )
        if self.ub is not None and self.ub.ndim != 3:
            raise ValueError(
                f"Upper bound must have shape (batch_size, n_constraints, 1). Got {self.ub.shape}."
            )

        if self.lb is not None and self.ub is not None:
            if self.lb.shape[1:] != self.ub.shape[1:]:
                raise ValueError(
                    f"Lower and upper bounds must have the same shape. Got {self.lb.shape} and {self.ub.shape}."
                )
            if (
                self.lb.shape[0] != self.ub.shape[0]
                and self.lb.shape[0] != 1
                and self.ub.shape[0] != 1
            ):
                raise ValueError(
                    f"Batch sizes of lower and upper bounds must match or one must be 1. Got {self.lb.shape} and {self.ub.shape}."
                )
            if not torch.all(self.lb <= self.ub):
                raise ValueError("Lower bound must be less than or equal to the upper bound.")

        if self.mask is not None:
            if self.mask.dtype != torch.bool:
                raise TypeError("Mask must be a boolean tensor.")
            if self.mask.ndim != 1:
                raise ValueError("Mask must be a 1D tensor.")

            dim = self.lb.shape if self.lb is not None else self.ub.shape
            if dim is not None and dim[1] != int(torch.sum(self.mask)):
                raise ValueError(
                    f"Number of active mask entries must match bounds. Got mask {self.mask.shape}, bounds {dim}."
                )


@dataclass(frozen=True)
class ProjectionInstance:
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x (torch.Tensor): The point to be projected.
            Shape (batch_size, dimension, 1)
        eq (Optional[EqualityConstraintsSpecification]): Equality constraint specs.
        box (Optional[BoxConstraintSpecification]): Box constraint specs.
    """

    x: torch.Tensor
    eq: Optional[EqualityConstraintsSpecification] = None
    box: Optional[BoxConstraintSpecification] = None

    def validate(self) -> None:
        """Validate the projection instance."""
        if self.x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch_size, dimension, 1). Got {self.x.shape}."
            )

    def update(self, **kwargs) -> "ProjectionInstance":
        """Return a new instance with updated attributes."""
        return replace(self, **kwargs)

    def to(self, device: torch.device) -> "ProjectionInstance":
        """Move tensors to a given device (e.g., GPU)."""
        return ProjectionInstance(
            x=self.x.to(device),
            eq=self.eq.to(device) if self.eq is not None else None,
            box=self.box.to(device) if self.box is not None else None,
        )


@dataclass(frozen=True)
class EquilibrationParams:
    """Dataclass for encapsulating equilibration parameters."""

    max_iter: int = 0
    tol: float = 1e-3
    ord: float = 2.0
    col_scaling: bool = False
    update_mode: str = "Gauss"
    safeguard: bool = False

    def validate(self) -> None:
        """Validate equilibration parameters."""
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if self.ord not in [1, 2, float("inf")]:
            raise ValueError("ord must be 1, 2, or infinity.")
        if self.update_mode not in ["Gauss", "Jacobi"]:
            raise ValueError('update_mode must be either "Gauss" or "Jacobi".')

    def update(self, **kwargs) -> "EquilibrationParams":
        """Return a new instance with updated attributes."""
        return replace(self, **kwargs)
