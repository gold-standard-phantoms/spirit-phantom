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
    "sha256:e4d76407070dd5932317bea4f93f47c9362aad5af05068dfc6f05a8dc4b254a1"
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
    "sha256:3b299c0ecbee0184ddd38e20f871a2152f8caaee1066117945bb708a73b9176f"
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
