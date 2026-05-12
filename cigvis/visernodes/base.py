from numbers import Real

import numpy as np
from matplotlib.colors import to_rgba

from cigvis import ExceptionWrapper

try:
    import viser
except BaseException as E:
    message = "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to enable viser"
    viser = ExceptionWrapper(E, message)


def as_scale3(scale):
    if isinstance(scale, Real):
        return [float(scale)] * 3
    scale = list(scale)
    if len(scale) != 3:
        raise ValueError(f"scale must be a scalar or length-3 sequence, got {scale}")
    return [float(s) for s in scale]


def color_to_uint8_rgb(color):
    if not isinstance(color, str):
        arr = np.asarray(color, dtype=np.float32)
        if arr.ndim == 1 and arr.size in (3, 4):
            arr = arr[:3]
            if arr.max() <= 1:
                arr = arr * 255
            return np.round(np.clip(arr, 0, 255)).astype(np.uint8)
    rgba = np.asarray(to_rgba(color), dtype=np.float32)
    return np.round(np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8)


def colors_to_uint8_rgb(colors):
    colors = np.asarray(colors)
    if colors.ndim == 1:
        return color_to_uint8_rgb(colors)
    if colors.shape[1] not in (3, 4):
        raise ValueError("colors must have shape (N, 3) or (N, 4)")
    colors = colors[:, :3].astype(np.float32)
    if colors.size and colors.max() <= 1:
        colors = colors * 255
    return np.round(np.clip(colors, 0, 255)).astype(np.uint8)


def auto_scale_from_points(points, target=1.5):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return [1.0, 1.0, 1.0]
    extent = np.ptp(points[:, :3], axis=0)
    max_extent = float(np.max(extent))
    if max_extent <= 0:
        max_extent = float(np.max(np.abs(points[:, :3])))
    if max_extent <= 0:
        return [1.0, 1.0, 1.0]
    return [target / max_extent] * 3


class ViserNodeMixin:

    def _init_node_state(self, name='node', scale=1.0):
        self._server = None
        self._name = name
        self._scale = as_scale3(scale)
        self.nodes = None

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
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)
        if self.server is not None:
            self.update_node()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = as_scale3(scale)
        self._on_scale_changed()
        if self.server is not None:
            self.update_node()

    def _on_scale_changed(self):
        pass

    @property
    def data_extent(self):
        return None

    def update_node(self):
        raise NotImplementedError("need to be implemented in subclass")
