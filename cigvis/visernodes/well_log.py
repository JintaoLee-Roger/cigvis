# Copyright (c) 2025 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from typing import List, Dict, Tuple, Union
import numpy as np
from cigvis import colormap, ExceptionWrapper
from .base import (
    ViserNodeMixin,
    auto_scale_from_points,
    color_to_uint8_rgb,
    colors_to_uint8_rgb,
)

try:
    import viser
except BaseException as E:
    message = "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to enable viser"
    viser = ExceptionWrapper(E, message)


class LogBase(ViserNodeMixin):

    def __init__(
        self,
        points: np.ndarray,
        values: np.ndarray = None,
        colors: np.ndarray = None,
        cmap: str = 'jet',
        clim: Tuple[float, float] = None,
        scale: int = -1,
    ):
        if scale < 0:
            scale = auto_scale_from_points(points, target=1.0)

        self._cmap = cmap
        self._clim = clim
        self.base_name = ''
        self._base_points = points.astype(np.float32)
        self._points = self._base_points
        self.points = self._base_points.copy()
        self._init_node_state('logs', scale)

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap
        self.update_node()

    @property
    def clim(self):
        return self._clim

    @clim.setter
    def clim(self, clim):
        self._clim = clim
        self.update_node()

    def update_node(self):
        raise NotImplementedError("need to be implemented in subclass")

    @property
    def wxyz(self):
        pass

    @property
    def position(self):
        pass

    @property
    def data_extent(self):
        return np.ptp(self._base_points[:, :3], axis=0)


class LogPoints(LogBase):

    def __init__(
        self,
        points: np.ndarray,
        values: np.ndarray = None,
        colors: np.ndarray = None,
        cmap: str = 'jet',
        clim: Tuple[float, float] = None,
        point_size: float = 1,
        point_shape: str = 'square',
        color=None,
        scale: int = -1,
        precision: str = 'float16',
        point_shading: str = 'gradient',
        **kwargs,
    ):
        super().__init__(points, values, colors, cmap, clim, scale)
        self._points = self._base_points
        self._values = values
        self._colors = colors
        self._color = color
        self.point_size = point_size
        self.point_shape = point_shape
        self.precision = precision
        self.point_shading = point_shading
        self.base_name = 'point'

    def process_points(self):
        self.points = self._points[:, :3] * np.asarray(self.scale)
        if self._color is not None:
            self.colors = np.tile(
                color_to_uint8_rgb(self._color),
                (len(self.points), 1),
            )
        elif self._colors is not None:
            self.colors = self._colors
        elif self._values is not None:
            clim = self.clim
            if clim is None:
                clim = [self._values.min(), self._values.max()]
            self.colors = colormap.get_colors_from_cmap(
                self.cmap,
                clim,
                self._values,
            )
        else:
            clim = self.clim
            if clim is None:
                clim = [self._points[:, 2].min(), self._points[:, 2].max()]
            self.colors = colormap.get_colors_from_cmap(
                self.cmap,
                clim,
                self._points[:, 2],
            )
        self.colors = colors_to_uint8_rgb(self.colors)

    def update_node(self):
        if self.server is None:
            return
        self.process_points()
        self.nodes = self.server.scene.add_point_cloud(
            self._name,
            self.points,
            self.colors,
            point_size=self.point_size * max(self.scale),
            point_shape=self.point_shape,
            precision=self.precision,
            point_shading=self.point_shading,
        )


class LogLineSegments(LogBase):

    def __init__(
        self,
        points: np.ndarray,
        values: np.ndarray = None,
        colors: np.ndarray = None,
        cmap: str = 'jet',
        clim: Tuple[float, float] = None,
        line_width: float = 1,
        scale: int = -1,
        **kwargs,
    ):
        super().__init__(points, values, colors, cmap, clim, scale)
        self._points = self._base_points
        self._values = values
        self._colors = colors
        self.line_width = line_width
        self.base_name = 'line'

    def process_points(self):
        scaled = self._points[:, :3] * np.asarray(self.scale)
        self.points = np.stack([scaled[:-1], scaled[1:]], axis=1)
        if self._colors is not None:
            self.colors = self._colors
        elif self._values is not None:
            clim = self.clim
            if clim is None:
                clim = [self._values.min(), self._values.max()]
            self.colors = colormap.get_colors_from_cmap(
                self.cmap,
                clim,
                self._values,
            )
        else:
            clim = self.clim
            if clim is None:
                clim = [self._points[:, 2].min(), self._points[:, 2].max()]
            self.colors = colormap.get_colors_from_cmap(
                self.cmap,
                clim,
                self._points[:, 2],
            )
        self.colors = colors_to_uint8_rgb(self.colors)
        self.colors = np.stack([self.colors[:-1], self.colors[:-1]], axis=1)

    def update_node(self):
        if self.server is None:
            return
        self.process_points()
        self.nodes = self.server.scene.add_line_segments(
            self._name,
            self.points,
            self.colors,
            line_width=self.line_width,
        )
