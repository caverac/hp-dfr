"""Utility functions for hp-dfr."""

from rich.console import Console

# Shared console for package output. Use markup=False for plain data logging
# (e.g. messages containing list reprs like "[16, 16]").
console = Console()

__all__: list[str] = ["console"]
