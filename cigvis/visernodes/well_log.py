# Copyright (c) 2025 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from typing import List, Dict, Tuple, Union
import numpy as np
from cigvis import colormap, ExceptionWrapper

try:
    import viser
except BaseException as E:
    message = "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to enable viser"
    viser = ExceptionWrapper(E, message)


class LogBase:

    def __init__(
        self,
        points: np.ndarray,
        values: np.ndarray = None,
        colors: np.ndarray = None,
        cmap: str = 'jet',
        clim: Tuple[float, float] = None,
        scale: int = -1,
    ):
        self._server = None  # viser.ViserServer
        if scale < 0:
            self._scale = [1 / points.max()] * 3
        else:
            self._scale = [scale] * 3

        self._cmap = cmap
        self._clim = clim
        self.base_name = ''
        self._points = points.astype(np.float32)
        self.points = points.astype(np.float32)

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, server):
        if not isinstance(server, viser.ViserServer):
            raise ValueError("server must be type: viser.ViserServer")
        self._server = server
        self.update_node()

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

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
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
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale
        self.update_node()


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
        scale: int = -1,
        **kwargs,
    ):
        super().__init__(points, values, colors, cmap, clim, scale)
        self._points = points
        self._values = values
        self._colors = colors
        self.point_size = point_size
        self.point_shape = point_shape
        self.base_name = 'point'

    def process_points(self):
        self.points[:, 0] = self._points[:, 0] * self.scale[0]
        self.points[:, 1] = self._points[:, 1] * self.scale[1]
        self.points[:, 2] = self._points[:, 2] * self.scale[2]
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
        self.colors = self.colors[:, :3]
        self.colors *= 255
        self.colors = self.colors.astype(np.uint8)

    def update_node(self):
        if self.server is None:
            return
        self.process_points()
        self.nodes = self.server.scene.add_point_cloud(
            self._name,
            self.points,
            self.colors,
            point_size=self.point_size * self.scale[2],
            point_shape=self.point_shape,
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
        self._points = points
        self._values = values
        self._colors = colors
        self.line_width = line_width
        self.base_name = 'line'

    def process_points(self):
        self.points = np.stack([self._points[:-1], self._points[1:]], axis=1)
        self.points[:, :, 0] *= self.scale[0]
        self.points[:, :, 1] *= self.scale[1]
        self.points[:, :, 2] *= self.scale[2]
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
        self.colors = self.colors[:, :3]
        self.colors *= 255
        self.colors = self.colors.astype(np.uint8)
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
