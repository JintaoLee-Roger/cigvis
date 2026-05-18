# Copyright (c) 2025 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
"""Internal node types for sliceviewer."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class MaskSpec:
    """An overlay volume attached to a SliceNode."""
    volume: np.ndarray
    cmap: object
    clim: Tuple[float, float]


@dataclass
class SliceNode:
    """Render a 2D frame from a 2D/3D/4D array."""

    volume: np.ndarray
    display_axes: Tuple[int, int]
    indices: Dict[int, int]
    cmap: object
    clim: Tuple[float, float]
    aspect: Union[str, float] = 1.0
    interpolation: Optional[Union[str, bool]] = "nearest"
    render_mode: str = "rgba"
    axis_labels: Tuple[str, ...] = field(default_factory=tuple)
    masks: List[MaskSpec] = field(default_factory=list)
    _linked: Optional['SliceNode'] = field(default=None, repr=False)

    @property
    def ndim(self) -> int:
        return self.volume.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.volume.shape)

    @property
    def y_axis(self) -> int:
        return self.display_axes[0]

    @property
    def x_axis(self) -> int:
        return self.display_axes[1]

    @property
    def hidden_axes(self) -> Tuple[int, ...]:
        return tuple(ax for ax in range(self.ndim) if ax not in self.display_axes)

    def axis_name(self, axis: int) -> str:
        return f"{self.axis_short_name(axis)} ({self.volume.shape[axis]})"

    def axis_short_name(self, axis: int) -> str:
        if self.axis_labels and axis < len(self.axis_labels):
            return self.axis_labels[axis]
        return f"dim {axis}"

    def clamp_axis_index(self, axis: int, idx: int) -> int:
        return int(np.clip(int(idx), 0, self.volume.shape[axis] - 1))

    def set_display_axes(self, display_axes: Tuple[int, int]) -> None:
        y_axis, x_axis = map(int, display_axes)
        if y_axis == x_axis:
            raise ValueError("display_axes must contain two different axes")
        if y_axis < 0 or y_axis >= self.ndim or x_axis < 0 or x_axis >= self.ndim:
            raise ValueError(f"display_axes must be valid for ndim={self.ndim}")
        self.display_axes = (y_axis, x_axis)
        self._ensure_indices()

    def set_index(self, axis: int, idx: int) -> None:
        if axis in self.display_axes:
            return
        self.indices[int(axis)] = self.clamp_axis_index(axis, idx)

    def get_index(self, axis: int) -> int:
        self._ensure_indices()
        return self.indices[int(axis)]

    def _ensure_indices(self) -> None:
        for axis in self.hidden_axes:
            if axis not in self.indices:
                self.indices[axis] = self.volume.shape[axis] // 2
            else:
                self.indices[axis] = self.clamp_axis_index(axis, self.indices[axis])
        for axis in list(self.indices):
            if axis in self.display_axes or axis >= self.ndim:
                self.indices.pop(axis, None)

    def get_frame(self) -> np.ndarray:
        return self._extract_frame(self.volume)

    def get_mask_frame(self, mask: MaskSpec) -> np.ndarray:
        return self._extract_frame(mask.volume)

    def _extract_frame(self, data: np.ndarray) -> np.ndarray:
        self._ensure_indices()
        selector = [slice(None)] * self.ndim
        for axis in self.hidden_axes:
            selector[axis] = self.get_index(axis)

        frame = data[tuple(selector)]
        current_axes = [axis for axis in range(self.ndim) if axis in self.display_axes]
        order = [current_axes.index(axis) for axis in self.display_axes]
        if order != list(range(len(order))):
            frame = np.transpose(frame, order)
        return frame

    def render(self) -> np.ndarray:
        """
        Blend base slice + all masks into an RGBA uint8 (h, w, 4) array.
        Uses cigvis.colormap.arrs_to_image, same as the 3D viewers.
        """
        from cigvis import colormap as cmap_mod

        frame = self.get_frame()
        arrs = [frame] + [self.get_mask_frame(m) for m in self.masks]
        cmaps = [self.cmap] + [m.cmap for m in self.masks]
        clims = [self.clim] + [m.clim for m in self.masks]
        rgba = cmap_mod.arrs_to_image(arrs, cmaps, clims, as_uint8=True)
        return rgba
