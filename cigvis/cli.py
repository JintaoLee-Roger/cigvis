"""Command line entry points for cigvis."""

from __future__ import annotations

from typing import Optional, Sequence


_REMOVED_MESSAGE = (
    "The standalone cigvis GUI has been removed. For existing VisPy nodes, "
    "use `cigvis.plot3D(..., gui=True)`. For SSH-friendly 2D slice viewing, "
    "install and use the `cigvis[sliceviewer]` extra."
)


def gui2d(argv: Optional[Sequence[str]] = None) -> int:
    raise SystemExit(_REMOVED_MESSAGE)


def gui3d(argv: Optional[Sequence[str]] = None) -> int:
    raise SystemExit(_REMOVED_MESSAGE)
