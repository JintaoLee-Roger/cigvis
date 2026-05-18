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


def _canvas_viewport(canvas) -> Tuple[int, int, int, int]:
    w, h = _normalize_render_size(canvas.physical_size, 'canvas.physical_size')
    return 0, 0, w, h


def _draw_screen_canvas(canvas, bgcolor) -> None:
    canvas.set_current()
    if hasattr(canvas, '_draw_scene'):
        canvas._draw_scene(bgcolor=bgcolor)
    else:
        canvas.context.clear(color=bgcolor, depth=True)


def _capture_screen_canvas(canvas, viewport, bgcolor) -> np.ndarray:
    _draw_screen_canvas(canvas, bgcolor)
    return _screenshot(viewport=viewport, alpha=True)


def _transparent_screen_canvas_image(canvas, viewport) -> np.ndarray:
    # Some VisPy image visuals keep framebuffer alpha at 0 on a transparent
    # clear color. Solve the visible RGBA from black/white opaque renders.
    black = _capture_screen_canvas(canvas, viewport, (0, 0, 0, 1))
    white = _capture_screen_canvas(canvas, viewport, (1, 1, 1, 1))

    black_rgb = black[..., :3].astype(np.float32) / 255.0
    white_rgb = white[..., :3].astype(np.float32) / 255.0

    alpha = 1.0 - np.max(white_rgb - black_rgb, axis=-1)
    alpha = np.clip(alpha, 0.0, 1.0)

    rgb = np.zeros_like(black_rgb)
    mask = alpha > (1.0 / 255.0)
    rgb[mask] = black_rgb[mask] / alpha[mask, None]
    rgb = np.clip(rgb, 0.0, 1.0)

    rgba = np.dstack((rgb, alpha))
    return np.round(rgba * 255.0).astype(np.uint8)


def _screen_canvas_image(canvas, transparent_bg: bool, bgcolor) -> np.ndarray:
    viewport = _canvas_viewport(canvas)
    if not transparent_bg:
        if bgcolor is not None:
            old_bgcolor = getattr(canvas, 'bgcolor', None)
            try:
                return _capture_screen_canvas(canvas, viewport, bgcolor)
            finally:
                if old_bgcolor is not None:
                    _draw_screen_canvas(canvas, old_bgcolor)
                canvas.update()
        return _screenshot(viewport=viewport, alpha=True)

    old_bgcolor = getattr(canvas, 'bgcolor', None)
    try:
        return _transparent_screen_canvas_image(canvas, viewport)
    finally:
        if old_bgcolor is not None:
            _draw_screen_canvas(canvas, old_bgcolor)
        canvas.update()


def _process_canvas_events(canvas) -> None:
    app = getattr(canvas, 'app', None)
    if app is None:
        return
    try:
        app.process_events()
    except Exception:
        pass


def _refresh_canvas_after_screenshot(canvas) -> None:
    try:
        canvas.update()
    except Exception:
        return
    _process_canvas_events(canvas)


def _canvas_visible(canvas) -> bool:
    backend = getattr(canvas, '_backend', None)
    is_visible = getattr(backend, 'isVisible', None)
    if callable(is_visible):
        try:
            return bool(is_visible())
        except Exception:
            return True
    return True


def _screen_framebuffer_canvas_image(canvas, transparent_bg: bool,
                                     bgcolor) -> np.ndarray:
    if not _canvas_visible(canvas):
        raise RuntimeError(
            "PNG save requires a visible VisPy canvas. Hidden/offscreen "
            "framebuffer export is not supported because VisPy's FBO render "
            "path is not equivalent to the displayed canvas."
        )
    try:
        return _screen_canvas_image(canvas, transparent_bg, bgcolor)
    finally:
        _refresh_canvas_after_screenshot(canvas)


def _save_canvas_png(canvas,
                     savename: str,
                     savedir: str = './',
                     save_kw: Dict = None) -> Path:
    save_kw = dict(save_kw or {})
    out = Path(savename)
    if not out.is_absolute():
        out = Path(savedir) / out
    out.parent.mkdir(parents=True, exist_ok=True)

    transparent_bg = bool(save_kw.pop('transparent_bg', True))
    bgcolor = save_kw.pop(
        'bgcolor',
        (0, 0, 0, 0) if transparent_bg else None,
    )
    if save_kw:
        unknown = ', '.join(sorted(save_kw))
        raise TypeError(f"Unknown save_kw parameter(s): {unknown}")

    image = _screen_framebuffer_canvas_image(canvas, transparent_bg, bgcolor)
    vispy.io.write_png(str(out), image)
    return out
