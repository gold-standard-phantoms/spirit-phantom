"""Tests for packaged SPIRIT fiducial loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spirit_phantom.core.fiducials import load_spirit_fiducials

if TYPE_CHECKING:
    from pathlib import Path


def test_load_spirit_fiducials_returns_34_unique_atlas_points() -> None:
    """Packaged fiducials should expose 34 unique atlas_mm landmarks."""
    fiducials = load_spirit_fiducials()
    assert len(fiducials) == 34
    assert len({fiducial.id for fiducial in fiducials}) == 34
    assert len({fiducial.confluence_address for fiducial in fiducials}) == 34
    assert all(len(fiducial.atlas_mm) == 3 for fiducial in fiducials)
    assert fiducials[0].id == "F01"
    assert fiducials[0].confluence_address == "C-5-t"


def test_load_spirit_fiducials_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Duplicate fiducial IDs in an override file must raise ValueError."""
    config_path = tmp_path / "fiducials.json"
    config_path.write_text(
        """
{
  "phantom": {"name": "SPIRIT", "part_number": "A-3011-0003", "issue": "1.0"},
  "fiducial_geometry": {
    "shape": "sphere",
    "diameter_mm": 8.0,
    "grid_spacing_mm": 34.0
  },
  "coordinate_frame": "atlas_mm",
  "fiducials": [
    {
      "id": "F01",
      "part": "lower_fiducial_grid",
      "confluence_address": "C-5-t",
      "component_label": 23,
      "layer": "inner",
      "atlas_mm": [0.0, 0.0, 0.0]
    },
    {
      "id": "F01",
      "part": "upper_fiducial_grid",
      "confluence_address": "C-5-v",
      "component_label": 24,
      "layer": "outer",
      "atlas_mm": [1.0, 0.0, 0.0]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate fiducial id"):
        load_spirit_fiducials(config_path=config_path)
