"""GUI colormap helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from matplotlib.colors import ListedColormap

from cigvis import colormap


LINE_CMAP_NAME = 'line_cmap'


@dataclass(frozen=True)
class GuiCmap:
    cmap: Any
    name: str
    is_line_cmap: bool = False


def is_line_cmap_expr(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    return text == LINE_CMAP_NAME or text.startswith(f'{LINE_CMAP_NAME}(')


def _literal_value(node):
    if not isinstance(node, ast.Constant):
        raise ValueError("line_cmap only accepts string and integer literals")
    value = node.value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("line_cmap only accepts string and integer literals")


def _parse_line_cmap(text: str, samples_hint: Optional[int] = None):
    text = text.strip()
    args = []
    kwargs = {}

    if text != LINE_CMAP_NAME:
        try:
            expr = ast.parse(text, mode='eval').body
        except SyntaxError as exc:
            raise ValueError(f"Invalid line_cmap expression: {text}") from exc
        if not (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == LINE_CMAP_NAME
        ):
            raise ValueError(f"Invalid line_cmap expression: {text}")
        args = [_literal_value(arg) for arg in expr.args]
        for keyword in expr.keywords:
            if keyword.arg not in {'cmap', 'n_lines', 'samples', 'seed'}:
                raise ValueError(f"Unsupported line_cmap argument: {keyword.arg}")
            kwargs[keyword.arg] = _literal_value(keyword.value)

    samples = int(samples_hint) if samples_hint else 256
    cmap = None
    n_lines = 20
    seed = 0

    if args:
        first = args[0]
        if isinstance(first, str):
            cmap = first
            numbers = args[1:]
        else:
            numbers = args
        if len(numbers) > 0:
            n_lines = int(numbers[0])
        if len(numbers) > 1:
            samples = int(numbers[1])
        if len(numbers) > 2:
            seed = int(numbers[2])

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    if 'n_lines' in kwargs:
        n_lines = int(kwargs['n_lines'])
    if 'samples' in kwargs:
        samples = int(kwargs['samples'])
    if 'seed' in kwargs:
        seed = int(kwargs['seed'])

    return cmap, n_lines, samples, seed


def _scale_existing_alpha(cmap, alpha: float):
    alpha = float(alpha)
    n = max(int(getattr(cmap, 'N', 256)), 2)
    colors = np.array(cmap(np.linspace(0, 1, n)), copy=True)
    nonzero = colors[:, 3] > 0
    colors[nonzero, 3] *= alpha
    return ListedColormap(colors, name=getattr(cmap, 'name', LINE_CMAP_NAME))


def resolve_gui_mpl_cmap(text: Any,
                         *,
                         samples_hint: Optional[int] = None,
                         alpha: Optional[float] = None,
                         excpt: str = 'None') -> GuiCmap:
    """Resolve editable GUI cmap text to a Matplotlib cmap."""
    if is_line_cmap_expr(text):
        cmap_name, n_lines, samples, seed = _parse_line_cmap(text, samples_hint)
        cmap_mpl = colormap.line_cmap(cmap_name,
                                      n_lines=n_lines,
                                      samples=samples,
                                      seed=seed)
        if alpha is not None:
            cmap_mpl = _scale_existing_alpha(cmap_mpl, alpha)
        return GuiCmap(
            cmap=cmap_mpl,
            name=str(text).strip(),
            is_line_cmap=True,
        )

    if alpha is None:
        return GuiCmap(
            cmap=colormap.cmap_to_mpl(text),
            name=str(text),
            is_line_cmap=False,
        )

    if excpt == 'None':
        cmap_mpl = colormap.set_alpha(text, alpha)
    elif excpt == 'min':
        cmap_mpl = colormap.set_alpha_except_min(text, alpha)
    elif excpt == 'max':
        cmap_mpl = colormap.set_alpha_except_max(text, alpha)
    elif excpt == 'ramp':
        cmap_mpl = colormap.ramp(text)
    else:
        raise ValueError(f"Unknown alpha exception mode: {excpt}")

    return GuiCmap(
        cmap=cmap_mpl,
        name=str(text),
        is_line_cmap=False,
    )


def resolve_gui_cmap(text: Any,
                     *,
                     samples_hint: Optional[int] = None,
                     alpha: Optional[float] = None,
                     excpt: str = 'None') -> GuiCmap:
    """Resolve editable GUI cmap text to a VisPy cmap."""
    resolved = resolve_gui_mpl_cmap(
        text,
        samples_hint=samples_hint,
        alpha=alpha,
        excpt=excpt,
    )
    return GuiCmap(
        cmap=colormap.cmap_to_vispy(resolved.cmap),
        name=resolved.name,
        is_line_cmap=resolved.is_line_cmap,
    )
