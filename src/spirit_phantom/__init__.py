"""Top-level package for `spirit_phantom`.

This module also exposes shared helpers for accessing the default SPIRIT atlas
via `pooch`. Both the CLI and tests should use the same constants and cache
behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pooch

DEFAULT_REGISTER_MOVING_IMAGE_URL = (
    "https://raw.githubusercontent.com/gold-standard-phantoms/public-data"
    "/main/phantoms/SPIRIT/atlas/"
    "spirit_issue1.0_vx0.25_sub2.nii.gz"
)
# Pinned URL for the default SPIRIT signal atlas (moving image for registration).

DEFAULT_REGISTER_MOVING_IMAGE_KNOWN_HASH: str | None = (
    "sha256:5d0614d32ec6c5b638db9b0f5e3a67d2e34765f5974d5a3568d5d9378e93ded0"
)
# Expected SHA-256 for the default SPIRIT atlas.
# If you need to update the atlas, update both the URL and this hash.

DEFAULT_SPIRIT_ATLAS_CACHE_NAMESPACE = "spirit-phantom"
# Pooch cache namespace used for the default SPIRIT atlas set (signal and component).

DEFAULT_REGISTER_MOVING_IMAGE_FILENAME = "spirit_issue1.0_vx0.25_sub2.nii.gz"
# Local filename used for the cached default SPIRIT atlas.


DEFAULT_COMPONENT_SPIRIT_ATLAS_IMAGE_URL = (
    "https://raw.githubusercontent.com/gold-standard-phantoms/public-data"
    "/main/phantoms/SPIRIT/atlas/spirit_issue1.0_vx0.25_sub2_components.nii.gz"
)
# Pinned URL for the default SPIRIT component atlas (moving image for ROI extraction).

DEFAULT_COMPONENT_SPIRIT_ATLAS_IMAGE_KNOWN_HASH: str | None = (
    "sha256:577e92b10e3855a8f93a89514f3eee79e2bc8917d3c6c861dba06c55433eef16"
)
# Expected SHA-256 for the default SPIRIT component atlas.
# If you need to update the atlas, update both the URL and this hash.

DEFAULT_COMPONENT_ATLAS_IMAGE_FILENAME = "spirit_issue1.0_vx0.25_sub2_components.nii.gz"
# Local filename used for the cached default SPIRIT component atlas.


def get_default_register_moving_image_path() -> Path:
    """Get the cached path to the default SPIRIT atlas moving image.

    The atlas is downloaded on first use and cached in the system cache for the
    configured namespace.

    Returns:
        Local path to the cached atlas image.
    """
    cache_directory = pooch.os_cache(DEFAULT_SPIRIT_ATLAS_CACHE_NAMESPACE)
    cached_path = pooch.retrieve(
        url=DEFAULT_REGISTER_MOVING_IMAGE_URL,
        known_hash=DEFAULT_REGISTER_MOVING_IMAGE_KNOWN_HASH,
        fname=DEFAULT_REGISTER_MOVING_IMAGE_FILENAME,
        path=cache_directory,
        progressbar=False,
    )
    return Path(cached_path)


def get_default_component_atlas_image_path() -> Path:
    """Get the cached path to the default SPIRIT component atlas image.

    The component atlas is downloaded on first use and cached in the system cache for the
    configured namespace.

    Returns:
        Local path to the cached component atlas image.
    """
    cache_directory = pooch.os_cache(DEFAULT_SPIRIT_ATLAS_CACHE_NAMESPACE)
    cached_path = pooch.retrieve(
        url=DEFAULT_COMPONENT_SPIRIT_ATLAS_IMAGE_URL,
        known_hash=DEFAULT_COMPONENT_SPIRIT_ATLAS_IMAGE_KNOWN_HASH,
        fname=DEFAULT_COMPONENT_ATLAS_IMAGE_FILENAME,
        path=cache_directory,
        progressbar=False,
    )
    return Path(cached_path)


def main() -> None:
    """Run the CLI entrypoint lazily."""
    from spirit_phantom.cli import main as cli_main  # noqa: PLC0415

    cli_main()


__all__ = [
    "DEFAULT_REGISTER_MOVING_IMAGE_FILENAME",
    "DEFAULT_REGISTER_MOVING_IMAGE_KNOWN_HASH",
    "DEFAULT_REGISTER_MOVING_IMAGE_URL",
    "DEFAULT_SPIRIT_ATLAS_CACHE_NAMESPACE",
    "get_default_component_atlas_image_path",
    "get_default_register_moving_image_path",
    "main",
]
