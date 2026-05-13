"""Command line entry points for cigvis."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def _build_gui3d_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open the modern PySide6 3D cigvis GUI.",
    )
    parser.add_argument("--nx", type=int, default=None, help="Initial x dimension.")
    parser.add_argument("--ny", type=int, default=None, help="Initial y dimension.")
    parser.add_argument("--nz", type=int, default=None, help="Initial z dimension.")
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="light",
        help="GUI theme.",
    )
    parser.add_argument(
        "--keep-dim",
        action="store_false",
        dest="clear_dim",
        help="Keep dimension fields after loading data.",
    )
    return parser


def gui3d(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_gui3d_parser().parse_args(argv)

    try:
        from cigvis.gui.gui3d import gui3d as launch_gui3d
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            raise SystemExit(
                "PySide6 is required for the modern GUI. Install it with "
                "`pip install \"cigvis[gui]\"` or `pip install PySide6`."
            ) from exc
        raise

    launch_gui3d(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        clear_dim=args.clear_dim,
        theme=args.theme,
    )
    return 0
