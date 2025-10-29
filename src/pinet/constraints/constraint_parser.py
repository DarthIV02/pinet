"""Parser of constraints to lifted representation module (PyTorch version)."""

from typing import Optional
import torch

from pinet.dataclasses import BoxConstraintSpecification
from .affine_equality import EqualityConstraint
from .affine_inequality import AffineInequalityConstraint
from .box import BoxConstraint


class ConstraintParser:
    """Parse constraints into a lifted representation.

    Converts equality, inequality, and box constraints into a unified lifted form
    suitable for optimization and projection methods.
    """

    def __init__(
        self,
        eq_constraint: Optional[EqualityConstraint],
        ineq_constraint: Optional[AffineInequalityConstraint],
        box_constraint: Optional[BoxConstraint] = None,
    ) -> None:
        """Initialize the constraint parser.

        Args:
            eq_constraint (EqualityConstraint): An equality constraint.
            ineq_constraint (AffineInequalityConstraint): An inequality constraint.
            box_constraint (BoxConstraint): A box constraint.
        """
        if ineq_constraint is None:
            # No lifting needed
            self.parse = lambda method=None: (eq_constraint, box_constraint, lambda y: y)
            return

        device = (
            ineq_constraint.C.device
            if hasattr(ineq_constraint, "C")
            else torch.device("cpu")
        )

        self.dim = ineq_constraint.dim

        # Default equality constraint if not provided
        if eq_constraint is None:
            eq_constraint = EqualityConstraint(
                A=torch.empty((1, 0, self.dim), device=device),
                b=torch.empty((1, 0, 1), device=device),
                method=None,
                var_b=False,
                var_A=False,
            )

        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint

        self.n_eq = eq_constraint.n_constraints
        self.n_ineq = ineq_constraint.n_constraints

        # Batch size consistency checks
        assert (
            eq_constraint.A.shape[0] == ineq_constraint.C.shape[0]
            or eq_constraint.A.shape[0] == 1
            or ineq_constraint.C.shape[0] == 1
        ), "Batch sizes of A and C must be consistent."

        if box_constraint is not None:
            assert (
                ineq_constraint.lb.shape[0] == box_constraint.lb.shape[0]
                or ineq_constraint.lb.shape[0] == 1
                or box_constraint.lb.shape[0] == 1
            ), "Batch sizes of lb and lower_bound must be consistent."
            assert (
                ineq_constraint.ub.shape[0] == box_constraint.ub.shape[0]
                or ineq_constraint.ub.shape[0] == 1
                or box_constraint.ub.shape[0] == 1
            ), "Batch sizes of ub and upper_bound must be consistent."

    def parse(
        self, method: Optional[str] = "pinv"
    ) -> tuple[EqualityConstraint, BoxConstraint, callable]:
        """Parse constraints into a lifted representation.

        Args:
            method (Optional[str]): Method used for solving linear systems ("pinv", None).

        Returns:
            tuple: (eq_lifted, box_lifted, lift_fn)
        """
        device = self.eq_constraint.A.device

        # Determine batch size for combining equality and inequality constraints
        mbAC = max(self.eq_constraint.A.shape[0], self.ineq_constraint.C.shape[0])

        # === Build lifted equality constraint ===
        first_row_batched = torch.tile(
            torch.cat(
                [
                    self.eq_constraint.A,
                    torch.zeros(
                        (self.eq_constraint.A.shape[0], self.n_eq, self.n_ineq),
                        device=device,
                    ),
                ],
                dim=2,
            ),
            (mbAC // self.eq_constraint.A.shape[0], 1, 1),
        )

        second_row_batched = torch.tile(
            torch.cat(
                [
                    self.ineq_constraint.C,
                    -torch.tile(
                        torch.eye(self.n_ineq, device=device).reshape(
                            1, self.n_ineq, self.n_ineq
                        ),
                        (self.ineq_constraint.C.shape[0], 1, 1),
                    ),
                ],
                dim=2,
            ),
            (mbAC // self.ineq_constraint.C.shape[0], 1, 1),
        )

        A_lifted = torch.cat([first_row_batched, second_row_batched], dim=1)
        b_lifted = torch.cat(
            [
                self.eq_constraint.b,
                torch.zeros(
                    (self.eq_constraint.b.shape[0], self.n_ineq, 1), device=device
                ),
            ],
            dim=1,
        )

        eq_lifted = EqualityConstraint(
            A=A_lifted,
            b=b_lifted,
            method=method,
            var_b=self.eq_constraint.var_b,
            var_A=self.eq_constraint.var_A,
        )

        # === Build lifted box constraint ===
        if self.box_constraint is None:
            # Only project lifted inequality part
            box_mask = torch.cat(
                [
                    torch.zeros(self.dim, dtype=torch.bool, device=device),
                    torch.ones(self.n_ineq, dtype=torch.bool, device=device),
                ]
            )

            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=self.ineq_constraint.lb,
                    ub=self.ineq_constraint.ub,
                    mask=box_mask,
                )
            )

        else:
            # Project both original and lifted parts
            box_mask = torch.cat(
                [
                    self.box_constraint.mask.to(device),
                    torch.ones(self.n_ineq, dtype=torch.bool, device=device),
                ]
            )

            mblb = max(
                self.box_constraint.lb.shape[0], self.ineq_constraint.lb.shape[0]
            )
            lifted_lb = torch.cat(
                [
                    torch.tile(
                        self.box_constraint.lb,
                        (mblb // self.box_constraint.lb.shape[0], 1, 1),
                    ),
                    torch.tile(
                        self.ineq_constraint.lb,
                        (mblb // self.ineq_constraint.lb.shape[0], 1, 1),
                    ),
                ],
                dim=1,
            )

            mbub = max(
                self.box_constraint.ub.shape[0], self.ineq_constraint.ub.shape[0]
            )
            lifted_ub = torch.cat(
                [
                    torch.tile(
                        self.box_constraint.ub,
                        (mbub // self.box_constraint.ub.shape[0], 1, 1),
                    ),
                    torch.tile(
                        self.ineq_constraint.ub,
                        (mbub // self.ineq_constraint.ub.shape[0], 1, 1),
                    ),
                ],
                dim=1,
            )

            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=lifted_lb,
                    ub=lifted_ub,
                    mask=box_mask,
                )
            )

        # === Define lifting function ===
        def lift(y):
            """Lift the input to the lifted dimension."""
            device = y.x.device
            y = y.update(
                x=torch.cat([y.x, self.ineq_constraint.C @ y.x], dim=1)
            )
            if self.eq_constraint.var_b:
                y = y.update(
                    eq=y.eq.update(
                        b=torch.cat(
                            [
                                y.eq.b,
                                torch.zeros((y.x.shape[0], self.n_ineq, 1), device=device),
                            ],
                            dim=1,
                        )
                    )
                )
            return y

        return (eq_lifted, box_lifted, lift)