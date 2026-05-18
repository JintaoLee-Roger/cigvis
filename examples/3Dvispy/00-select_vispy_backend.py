# Copyright (c) 2026 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Select the VisPy backend
========================

VisPy selects the first usable GUI backend it finds. When multiple GUI
toolkits are installed, such as PyQt5 and PySide6, choose the backend before
importing ``cigvis`` or creating any VisPy canvas:

.. code-block:: python

    import vispy
    vispy.use(app="pyside6")  # or "pyqt5", "pyqt6", "glfw", ...

    import cigvis

For plain ``cigvis.plot3D(..., gui=False)``, any working VisPy backend can be
used. For ``cigvis.plot3D(..., gui=True)``, use ``pyside6``. The GUI inspector
is a PySide6 Qt window, so mixing it with a previously selected PyQt backend is
not supported.

Run this file to inspect or select the backend without opening a window:

.. code-block:: bash

    python examples/3Dvispy/00-select_vispy_backend.py
    python examples/3Dvispy/00-select_vispy_backend.py pyside6
    python examples/3Dvispy/00-select_vispy_backend.py pyside6 --gui
"""

import argparse

import vispy
from vispy import app as vispy_app


def main():
    parser = argparse.ArgumentParser(
        description="Inspect or choose the VisPy application backend.")
    parser.add_argument(
        "backend",
        nargs="?",
        help="VisPy backend name, e.g. pyside6, pyqt5, pyqt6, glfw, or sdl2.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Check the backend for cigvis.plot3D(..., gui=True).")
    args = parser.parse_args()

    if args.gui and args.backend and args.backend.lower() != "pyside6":
        parser.error("cigvis.plot3D(..., gui=True) requires the pyside6 backend")

    if args.backend:
        vispy.use(app=args.backend)

    app = vispy_app.use_app()
    print(f"VisPy backend: {app.backend_name}")

    if args.gui and app.backend_name.lower() != "pyside6":
        raise SystemExit("cigvis.plot3D(..., gui=True) requires pyside6")


if __name__ == "__main__":
    main()
