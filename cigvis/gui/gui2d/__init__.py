"""Compatibility stub for the removed standalone 2D GUI."""

_REMOVED_MESSAGE = (
    "The standalone cigvis 2D GUI has been removed. For SSH-friendly 2D "
    "slice viewing, install and use the `cigvis[sliceviewer]` extra."
)


def gui2d(*_args, **_kwargs):
    raise RuntimeError(_REMOVED_MESSAGE)


Gui2dWindow = None

__all__ = ["gui2d", "Gui2dWindow"]
