"""Typed override models for registration parameter optimisation.

These models provide a curated and explicit surface for stage-level elastix
parameter overrides. Only non-``None`` values are applied.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _BaseStageOverrides:
    """Base class for curated stage-level elastix parameter overrides."""

    MaximumNumberOfIterations: int | None = None

    def to_elastix_updates(self) -> dict[str, tuple[str, ...]]:
        """Build parameter updates in elastix map format.

        Returns:
            A mapping from elastix parameter name to tuple-based values.

        Raises:
            ValueError: If any provided override value is invalid.
        """
        updates: dict[str, tuple[str, ...]] = {}
        if self.MaximumNumberOfIterations is not None:
            if self.MaximumNumberOfIterations <= 0:
                msg = "MaximumNumberOfIterations must be a positive integer."
                raise ValueError(msg)
            updates["MaximumNumberOfIterations"] = (
                str(self.MaximumNumberOfIterations),
            )
        return updates


@dataclass(slots=True)
class RigidOverrides(_BaseStageOverrides):
    """Curated overrides for the rigid registration stage."""


@dataclass(slots=True)
class AffineOverrides(_BaseStageOverrides):
    """Curated overrides for the affine registration stage."""


@dataclass(slots=True)
class BSplineOverrides(_BaseStageOverrides):
    """Curated overrides for the B-spline registration stage."""


@dataclass(slots=True)
class RegistrationOverrides:
    """Container for optional stage-level registration overrides."""

    RigidOverrides: RigidOverrides | None = None
    AffineOverrides: AffineOverrides | None = None
    BSplineOverrides: BSplineOverrides | None = None
