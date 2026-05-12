# Copyright (c) 2026 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
"""PNG export helpers shared by VisPy canvas saves and keyboard shortcuts."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import vispy
from vispy.gloo.util import _screenshot


def _normalize_render_size(size, name: str = 'size') -> Tuple[int, int]:
    if isinstance(size, str):
        parts = size.lower().replace('x', ',').split(',')
        if len(parts) != 2:
            raise ValueError(f"{name} must look like '3000x2000'")
        size = (parts[0], parts[1])
    if len(size) != 2:
        raise ValueError(f"{name} must contain two values")
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} values must be positive")
    return w, h


def _fit_render_size(target_size: Tuple[int, int],
                     source_size: Tuple[int, int],
                     output_policy: str) -> Tuple[int, int]:
    tw, th = target_size
    sw, sh = source_size
    if output_policy == 'fit':
        scale = max(tw, th) / max(sw, sh)
        return max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    if output_policy == 'pad':
        source_aspect = sw / sh
        target_aspect = tw / th
        if target_aspect > source_aspect:
            return max(1, int(round(th * source_aspect))), th
        return tw, max(1, int(round(tw / source_aspect)))
    raise ValueError("save_kw['output_policy'] must be 'fit' or 'pad'")


def _color_to_ubyte(color, channels: int) -> np.ndarray:
    rgba = np.asarray(vispy.color.Color(color).rgba)
    values = np.clip(np.round(rgba * 255), 0, 255).astype(np.uint8)
    if channels == 3:
        return values[:3]
    return values


def _compose_rendered_image(image: np.ndarray,
                            target_size: Tuple[int, int],
                            output_policy: str,
                            pad_color='white',
                            transparent_bg: bool = False) -> np.ndarray:
    tw, th = target_size
    h, w = image.shape[:2]
    if output_policy == 'fit' or (w, h) == target_size:
        return image
    if output_policy != 'pad':
        raise ValueError("save_kw['output_policy'] must be 'fit' or 'pad'")

    out = np.empty((th, tw, image.shape[2]), dtype=image.dtype)
    if transparent_bg and image.shape[2] == 4:
        out[...] = 0
    else:
        out[...] = _color_to_ubyte(pad_color, image.shape[2])

    x0 = max(0, (tw - w) // 2)
    y0 = max(0, (th - h) // 2)
    out[y0:y0 + h, x0:x0 + w] = image
    return out


def _canvas_viewport(canvas) -> Tuple[int, int, int, int]:
    w, h = _normalize_render_size(canvas.physical_size, 'canvas.physical_size')
    return 0, 0, w, h


def _draw_screen_canvas(canvas, bgcolor) -> None:
    canvas.set_current()
    if hasattr(canvas, '_draw_scene'):
        canvas._draw_scene(bgcolor=bgcolor)
    else:
        canvas.context.clear(color=bgcolor, depth=True)


def _screen_canvas_image(canvas, transparent_bg: bool, bgcolor) -> np.ndarray:
    viewport = _canvas_viewport(canvas)
    if not transparent_bg:
        return _screenshot(viewport=viewport, alpha=True)

    old_bgcolor = getattr(canvas, 'bgcolor', None)
    _draw_screen_canvas(canvas, bgcolor)
    try:
        return _screenshot(viewport=viewport, alpha=True)
    finally:
        if old_bgcolor is not None:
            _draw_screen_canvas(canvas, old_bgcolor)
        canvas.update()


def _save_canvas_png(canvas,
                     savename: str,
                     savedir: str = './',
                     save_kw: Dict = None) -> Path:
    save_kw = dict(save_kw or {})
    mode = save_kw.pop('mode', 'offscreen')
    if mode not in ('offscreen', 'screen'):
        raise ValueError("save_kw['mode'] must be 'offscreen' or 'screen'")

    out = Path(savename)
    if not out.is_absolute():
        out = Path(savedir) / out
    out.parent.mkdir(parents=True, exist_ok=True)

    canvas_size = _normalize_render_size(canvas.size, 'canvas.size')
    transparent_bg = bool(save_kw.pop('transparent_bg', False))
    bgcolor = save_kw.pop(
        'bgcolor',
        (0, 0, 0, 0) if transparent_bg else None,
    )

    if mode == 'screen':
        save_kw.pop('size', None)
        save_kw.pop('output_policy', None)
        save_kw.pop('policy', None)
        save_kw.pop('pad_color', None)
        if save_kw:
            unknown = ', '.join(sorted(save_kw))
            raise TypeError(f"Unknown save_kw parameter(s): {unknown}")
        image = _screen_canvas_image(canvas, transparent_bg, bgcolor)
        vispy.io.write_png(str(out), image)
        return out

    target_size = save_kw.pop('size', None)
    target_size = _normalize_render_size(target_size,
                                         "save_kw['size']") if target_size is not None else canvas_size
    output_policy = save_kw.pop('output_policy', save_kw.pop('policy', 'fit'))
    pad_color = save_kw.pop('pad_color', 'white')
    if save_kw:
        unknown = ', '.join(sorted(save_kw))
        raise TypeError(f"Unknown save_kw parameter(s): {unknown}")

    render_size = _fit_render_size(target_size, canvas_size, output_policy)
    old_physical_size = tuple(int(v) for v in canvas.physical_size)
    if hasattr(canvas._backend, '_vispy_set_physical_size'):
        canvas._backend._vispy_set_physical_size(*render_size)

    try:
        render_kwargs = {
            'region': (0, 0, canvas_size[0], canvas_size[1]),
            'size': render_size,
            'alpha': transparent_bg,
        }
        if bgcolor is not None:
            render_kwargs['bgcolor'] = bgcolor
        image = canvas.render(**render_kwargs)
    finally:
        if hasattr(canvas._backend, '_vispy_set_physical_size'):
            canvas._backend._vispy_set_physical_size(*old_physical_size)

    image = _compose_rendered_image(
        image,
        target_size,
        output_policy,
        pad_color=pad_color,
        transparent_bg=transparent_bg,
    )
    vispy.io.write_png(str(out), image)
    return out
