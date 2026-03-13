"""Top-level package for spirit_phantom."""


def main() -> None:
    """Run the CLI entrypoint lazily."""
    from spirit_phantom.cli import main as cli_main  # noqa: PLC0415

    cli_main()


__all__ = ["main"]
