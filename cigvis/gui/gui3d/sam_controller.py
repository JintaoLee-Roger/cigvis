"""SAM-like prompt controller used by the PySide6 3D GUI."""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import vispy

from cigvis import colormap
from cigvis.vispynodes.axis_aligned_image import AxisAlignedImage
from cigvis.vispynodes.volume_image import VolumeImage


@dataclass
class Prompt:
    xyz: Tuple[int, int, int]
    axis: str
    slice_pos: int
    r: int
    value: int = 250


@dataclass
class SamControllerConfig:
    mask_overlay_name: str = "fault"
    prompt_overlay_name: str = "prompt"
    poll_hz: float = 30.0
    max_workers: int = 1
    prompt_value: int = 250
    prompt_radius_min: int = 0
    prompt_radius_max: int = 3
    prompt_radius_ratio: float = 0.0001


class SamLikeController:
    """Connect Alt-click prompts on ``VisCanvas`` to a user supplied decoder."""

    def __init__(
        self,
        vol: VolumeImage,
        canvas,
        decode_fn: Optional[Callable] = None,
        on_prompt_count: Optional[Callable[[int], None]] = None,
        cfg: Optional[SamControllerConfig] = None,
    ) -> None:
        self.vol = vol
        self.canvas = canvas
        self.decode_fn = decode_fn
        self.on_prompt_count = on_prompt_count
        self.cfg = cfg or SamControllerConfig()

        self.prompt_xyz: Optional[Tuple[int, int, int]] = None
        self.prompts: List[Prompt] = []
        self.prompt_vol = np.zeros_like(self.vol.volume, dtype=np.uint8)

        prompt_cmap = colormap.set_alpha_except_min('jet', alpha=1.0)
        if self.cfg.prompt_overlay_name in getattr(self.vol, "_overlays", {}):
            self.vol.replace_overlay_volume(
                self.cfg.prompt_overlay_name,
                self.prompt_vol,
                refresh=True,
            )
        else:
            self.vol.add_overlay_volume(
                name=self.cfg.prompt_overlay_name,
                volume=self.prompt_vol,
                cmap=prompt_cmap,
                interpolation='nearest',
            )

        self._executor = ThreadPoolExecutor(max_workers=self.cfg.max_workers)
        self._result_q: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue()
        self._job_id = 0
        self._running = False

        self._poll_timer = vispy.app.Timer(
            interval=1.0 / self.cfg.poll_hz,
            connect=self._poll_results,
            start=True,
        )

        self.canvas.set_prompt_callback(self._on_prompt_pick)
        self.canvas.events.key_press.connect(self._on_key_press)
        self._emit_prompt_count()

    def close(self) -> None:
        try:
            self._poll_timer.stop()
        except Exception:
            pass
        try:
            self.canvas.events.key_press.disconnect(self._on_key_press)
        except Exception:
            pass
        try:
            self.canvas.set_prompt_callback(None)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=False)

    def submit_decode(self, xyz: Tuple[int, int, int]) -> None:
        if self._running:
            return
        if self.decode_fn is None:
            return

        self._job_id += 1
        job_id = self._job_id
        self._running = True
        base_vol = self.vol.volume
        prompts = list(self.prompts)

        def _worker():
            mask = self.decode_fn(base_vol, prompts, xyz)
            self._result_q.put((job_id, mask))

        self._executor.submit(_worker)

    def _on_key_press(self, event) -> None:
        if event.key is None:
            return
        key_name = event.key.name.lower()
        if key_name in ("enter", "return"):
            if self.prompt_xyz is not None:
                self.submit_decode(self.prompt_xyz)
            return
        if key_name in ("backspace", "delete") and self.prompts:
            self._undo_last_prompt()
            return
        if event.text == 'c':
            self._clear_prompts()

    def _on_prompt_pick(self, xyz, hover_on=None, event=None) -> None:
        img = hover_on
        if img is None:
            return

        x, y, z = [int(np.round(v)) for v in xyz]
        x = max(0, min(self.vol.shape[0] - 1, x))
        y = max(0, min(self.vol.shape[1] - 1, y))
        z = max(0, min(self.vol.shape[2] - 1, z))
        self.prompt_xyz = (x, y, z)

        pr = Prompt(
            xyz=self.prompt_xyz,
            axis=img.axis,
            slice_pos=int(img.pos),
            r=self._auto_prompt_radius(img),
            value=self.cfg.prompt_value,
        )
        self.prompts.append(pr)
        self._apply_prompt_to_volume(pr, op="add")
        self._refresh_prompt_overlay_on_node(img)
        self._emit_prompt_count()

    def _auto_prompt_radius(self, img: AxisAlignedImage) -> int:
        h, w = img.image_funcs[0](img.pos, get_shape=True)
        base = int(round(min(h, w) * self.cfg.prompt_radius_ratio))
        return max(self.cfg.prompt_radius_min,
                   min(self.cfg.prompt_radius_max, base))

    def _apply_prompt_to_volume(self, pr: Prompt, op: str = "add") -> None:
        x, y, z = pr.xyz
        r = pr.r
        ni, nx, nt = self.prompt_vol.shape
        value = pr.value if op == "add" else 0

        def clamp(a, lo, hi):
            return max(lo, min(hi, a))

        if pr.axis == 'z':
            x0, x1 = clamp(x - r, 0, ni - 1), clamp(x + r, 0, ni - 1)
            y0, y1 = clamp(y - r, 0, nx - 1), clamp(y + r, 0, nx - 1)
            self.prompt_vol[x0:x1 + 1, y0:y1 + 1, z] = value
        elif pr.axis == 'y':
            x0, x1 = clamp(x - r, 0, ni - 1), clamp(x + r, 0, ni - 1)
            z0, z1 = clamp(z - r, 0, nt - 1), clamp(z + r, 0, nt - 1)
            self.prompt_vol[x0:x1 + 1, y, z0:z1 + 1] = value
        else:
            y0, y1 = clamp(y - r, 0, nx - 1), clamp(y + r, 0, nx - 1)
            z0, z1 = clamp(z - r, 0, nt - 1), clamp(z + r, 0, nt - 1)
            self.prompt_vol[x, y0:y1 + 1, z0:z1 + 1] = value

    def _rebuild_prompt_volume(self) -> None:
        self.prompt_vol.fill(0)
        for pr in self.prompts:
            self._apply_prompt_to_volume(pr, op="add")

    def _undo_last_prompt(self) -> None:
        self.prompts.pop()
        self.prompt_xyz = self.prompts[-1].xyz if self.prompts else None
        self._rebuild_prompt_volume()
        self._refresh_prompt_overlay_all()
        self._emit_prompt_count()

    def _clear_prompts(self) -> None:
        self.prompts.clear()
        self.prompt_xyz = None
        self.prompt_vol.fill(0)
        self._refresh_prompt_overlay_all()
        self._emit_prompt_count()

    def _refresh_prompt_overlay_on_node(self, img: AxisAlignedImage) -> None:
        refresh_one = getattr(self.vol, "refresh_overlay_on_node", None)
        if refresh_one is not None:
            refresh_one(self.cfg.prompt_overlay_name, img)
        else:
            self._refresh_prompt_overlay_all()

    def _refresh_prompt_overlay_all(self) -> None:
        self.vol.refresh_overlay(self.cfg.prompt_overlay_name)
        try:
            self.canvas.update()
        except Exception:
            pass

    def _poll_results(self, event=None) -> None:
        applied = False
        while True:
            try:
                job_id, mask = self._result_q.get_nowait()
            except queue.Empty:
                break
            if job_id != self._job_id:
                continue
            mask = np.asarray(mask)
            if self.cfg.mask_overlay_name in getattr(self.vol, "_overlays", {}):
                self.vol.replace_overlay_volume(
                    self.cfg.mask_overlay_name,
                    mask,
                    refresh=True,
                )
            else:
                self.vol.add_overlay_volume(
                    name=self.cfg.mask_overlay_name,
                    volume=mask,
                    cmap=colormap.set_alpha_except_min('jet', alpha=1.0),
                    interpolation='nearest',
                )
            self._running = False
            applied = True

        if applied:
            try:
                self.canvas.update()
            except Exception:
                pass

    def _emit_prompt_count(self) -> None:
        if self.on_prompt_count is not None:
            self.on_prompt_count(len(self.prompts))
