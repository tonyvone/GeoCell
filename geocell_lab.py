"""Backwards-compatible launcher: `python geocell_lab.py` starts the REPL."""
from geocell.cli import cli

if __name__ == "__main__":
    cli()
