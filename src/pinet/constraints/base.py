"""Abstract class for constraint sets (PyTorch version)."""

from abc import ABC, abstractmethod
import torch

from pinet.dataclasses import ProjectionInstance


class Constraint(ABC):
    """Abstract base class for constraint sets."""

    @abstractmethod
    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to project.

        Returns:
            ProjectionInstance: The projected input.
        """
        pass

    @abstractmethod
    def cv(self, yraw: ProjectionInstance) -> torch.Tensor:
        """Compute the constraint violation.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            torch.Tensor: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Returns:
            int: The dimension of the constraint set.
        """
        pass

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """Return the number of constraints.

        Returns:
            int: The number of constraints.
        """
        pass