"""Top-level package for `spirit_phantom`.

This module also exposes shared helpers for accessing the default SPIRIT atlas
via `pooch`. Both the CLI and tests should use the same constants and cache
behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pooch

DEFAULT_REGISTER_MOVING_IMAGE_URL = (
    "https://raw.githubusercontent.com/gold-standard-phantoms/public-data/"
    "e01d6b8e642e679257aa1d6c42816754bc2cc228/phantoms/SPIRIT/atlas/"
    "spirit_issue1.0_vx0.25_sub1.nii.gz"
)
# Pinned URL for the default SPIRIT atlas (moving image).

DEFAULT_REGISTER_MOVING_IMAGE_KNOWN_HASH: str | None = (
    "sha256:77f027524325c4ad4d2d23ee8c224dfd04080216531c309d93e9fe41d686739d"
)
# Expected SHA-256 for the default SPIRIT atlas.
# If you need to update the atlas, update both the URL and this hash.

DEFAULT_REGISTER_MOVING_IMAGE_CACHE_NAMESPACE = "spirit-phantom"
# Pooch cache namespace used for the default SPIRIT atlas.

DEFAULT_REGISTER_MOVING_IMAGE_FILENAME = "spirit_issue1.0_vx0.25_sub1.nii.gz"
# Local filename used for the cached default SPIRIT atlas.


def get_default_register_moving_image_path() -> Path:
    """Get the cached path to the default SPIRIT atlas moving image.

    The atlas is downloaded on first use and cached in the system cache for the
    configured namespace.

    Returns:
        Local path to the cached atlas image.
    """
    cache_directory = pooch.os_cache(DEFAULT_REGISTER_MOVING_IMAGE_CACHE_NAMESPACE)
    cached_path = pooch.retrieve(
        url=DEFAULT_REGISTER_MOVING_IMAGE_URL,
        known_hash=DEFAULT_REGISTER_MOVING_IMAGE_KNOWN_HASH,
        fname=DEFAULT_REGISTER_MOVING_IMAGE_FILENAME,
        path=cache_directory,
        progressbar=False,
    )
    return Path(cached_path)


def main() -> None:
    """Run the CLI entrypoint lazily."""
    from spirit_phantom.cli import main as cli_main  # noqa: PLC0415

    cli_main()


__all__ = [
    "DEFAULT_REGISTER_MOVING_IMAGE_CACHE_NAMESPACE",
    "DEFAULT_REGISTER_MOVING_IMAGE_FILENAME",
    "DEFAULT_REGISTER_MOVING_IMAGE_KNOWN_HASH",
    "DEFAULT_REGISTER_MOVING_IMAGE_URL",
    "get_default_register_moving_image_path",
    "main",
]
