"""Packaged SPIRIT phantom fiducial landmark coordinates.

Provides strongly typed access to the 34 spherical fiducial landmarks of the
SPIRIT phantom (part number A-3011-0003, issue 1.0). All published SPIRIT atlas
resolutions share the same NIfTI world millimetre frame (``atlas_mm``), centred
on the phantom's central plate, so only ``atlas_mm`` is needed for
physical-space registration and distortion analyses.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

FIDUCIAL_CONFIG_FILENAME = "fiducials.json"
FIDUCIAL_CONFIG_PATH = (
    Path(__file__).parent / "configuration" / FIDUCIAL_CONFIG_FILENAME
)

FiducialPart = Literal["lower_fiducial_grid", "upper_fiducial_grid"]
FiducialLayer = Literal["inner", "outer"]


class Fiducial(BaseModel):
    """Single SPIRIT spherical fiducial landmark in atlas world millimetres.

    Attributes:
        id: Unique fiducial identifier (``F01`` to ``F34``).
        part: Physical grid assembly (``lower_fiducial_grid`` or
            ``upper_fiducial_grid``).
        confluence_address: Letter-number-letter address from the SPIRIT
            Fiducial label scheme Confluence page (for example ``C-5-t``).
        component_label: Label value in the SPIRIT component atlas
            (23 for lower, 24 for upper).
        layer: Axial layer within its grid (``inner`` at ±34 mm or ``outer`` at
            ±68 mm).
        atlas_mm: Sphere centre ``(x, y, z)`` in NIfTI world millimetres of the
            published SPIRIT atlas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    part: FiducialPart
    confluence_address: str
    component_label: int
    layer: FiducialLayer
    atlas_mm: tuple[float, float, float]

    @field_validator("confluence_address")
    @classmethod
    def _validate_confluence_address(cls, value: str) -> str:
        """Ensure the Confluence address is a non-empty letter-number-letter code."""
        stripped = value.strip()
        if not stripped:
            msg = "Fiducial confluence_address must be non-empty."
            raise ValueError(msg)
        return stripped

    @field_validator("atlas_mm")
    @classmethod
    def _validate_finite_coords(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Ensure all three coordinates are finite numbers."""
        if not all(math.isfinite(coord) for coord in value):
            msg = f"Fiducial atlas_mm coordinates must be finite, got {value}."
            raise ValueError(msg)
        return value


class _PhantomMetadata(BaseModel):
    """Phantom identification metadata in the fiducials configuration file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    part_number: str
    issue: str


class _FiducialGeometry(BaseModel):
    """Physical geometry metadata for the fiducial spheres."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shape: str
    diameter_mm: float
    grid_spacing_mm: float


class _FiducialConfigurationPayload(BaseModel):
    """Root schema for ``fiducials.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phantom: _PhantomMetadata
    fiducial_geometry: _FiducialGeometry
    coordinate_frame: str
    fiducials: list[Fiducial]


def _read_default_fiducials_json() -> str:
    """Read the packaged ``fiducials.json`` text from package data or source.

    Returns:
        Raw UTF-8 JSON content.

    Raises:
        FileNotFoundError: If ``fiducials.json`` cannot be located.
    """
    try:
        config_package = resources.files("spirit_phantom.core.configuration")
        config_resource = config_package / FIDUCIAL_CONFIG_FILENAME
        if config_resource.is_file():
            return config_resource.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    if not FIDUCIAL_CONFIG_PATH.is_file():
        msg = (
            f"Fiducial configuration file '{FIDUCIAL_CONFIG_FILENAME}' not found. "
            f"Checked package data and source location: {FIDUCIAL_CONFIG_PATH}"
        )
        raise FileNotFoundError(msg)
    return FIDUCIAL_CONFIG_PATH.read_text(encoding="utf-8")


def _validate_unique_fiducial_ids(*, fiducials: list[Fiducial]) -> list[Fiducial]:
    """Ensure fiducial list is non-empty and all fiducial IDs are unique.

    Args:
        fiducials: Parsed fiducial entries.

    Returns:
        Validated fiducial list.

    Raises:
        ValueError: If ``fiducials`` is empty or contains duplicate IDs.
    """
    if not fiducials:
        msg = "Fiducial configuration contains no fiducials."
        raise ValueError(msg)

    seen_ids: set[str] = set()
    for fiducial in fiducials:
        if fiducial.id in seen_ids:
            msg = f"Duplicate fiducial id '{fiducial.id}' in fiducial configuration."
            raise ValueError(msg)
        seen_ids.add(fiducial.id)
    return fiducials


def load_spirit_fiducials(*, config_path: Path | None = None) -> tuple[Fiducial, ...]:
    """Load the SPIRIT phantom fiducial coordinates in atlas world millimetres.

    Args:
        config_path: Optional override path to a fiducials JSON file. When
            omitted, loads the packaged ``fiducials.json``.

    Returns:
        Immutable tuple of validated :class:`Fiducial` records in file order.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the JSON payload is invalid, empty, or has duplicate IDs.
    """
    if config_path is not None:
        if not config_path.is_file():
            msg = f"Fiducial configuration file not found: {config_path}"
            raise FileNotFoundError(msg)
        raw_text = config_path.read_text(encoding="utf-8")
    else:
        raw_text = _read_default_fiducials_json()

    try:
        raw_payload: Any = json.loads(raw_text)
        payload = _FiducialConfigurationPayload.model_validate(raw_payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        msg = "Invalid SPIRIT fiducials JSON structure."
        raise ValueError(msg) from exc

    validated = _validate_unique_fiducial_ids(fiducials=payload.fiducials)
    return tuple(validated)
