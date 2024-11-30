# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Based on the `Mesh` of vispy, we create a new class to maintenance the instance of surface, instead of the general `Mesh` class 
"""

from typing import Callable, List, Tuple
import warnings
import numpy as np
from vispy.scene import Mesh
import cigvis
from cigvis import colormap
from cigvis.meshs import surface2mesh, arbline2mesh
from cigvis.utils import surfaceutils
from .axis_aligned_image import AxisAlignedImage
import matplotlib.colors as mcolors


class SurfaceNode(Mesh):

    def __init__(self,
                 surf,
                 volume=None,
                 values='depth',
                 clims: List = None,
                 cmaps='jet',
                 shape: Tuple = None,
                 step1: int = 2,
                 step2: int = 2,
                 offset: List = [0, 0, 0],
                 interval: List = [1, 1, 1],
                 interp: bool = True,
                 anti_rot: bool = True,
                 shading: str = 'smooth',
                 dyn_light: bool = True,
                 **kwargs):
        self.volume = volume
        self.steps = [int(step1), int(step2)]
        self.anti_rot = anti_rot
        self.orig_surf = surf
        self.surf = None
        self.values = values
        self._offset = np.array(offset)
        self._interval = np.array(interval)
        self.seis_value = None  # only interp once
        # to deal with multiple values
        self._cmaps = cmaps
        self._clims = clims
        self.is_instance = False
        self._interp = interp
        self.render_type = -1  # 0 for colors, 1 for vetex_values, and 2 for vetex_colors
        self.dyn_light = dyn_light

        # define shape
        if volume is None and shape is None:
            if surf.shape[1] == 3:
                raise ValueError(
                    "must pass at least one parameters of `volume`, `shape`")
            else:
                shape = surf.shape
        if volume is not None:
            shape = volume.shape
            if cigvis.is_line_first():
                self.shape = shape[:2]
            else:
                self.shape = shape[1:]
        else:
            self.shape = shape

        vertices, faces = self.to_meshs()

        super().__init__(vertices=vertices,
                         faces=faces,
                         shading=shading,
                         **kwargs)
        self.process_values()

        self.is_instance = True

    def to_meshs(self, method='cubic', fill=-1):
        """
        surf: shape as (n1, n2), or (N, 3)
        """
        surf = self.orig_surf
        assert surf.ndim == 2, f"surface's shape must be (ni, nx), or (N, 3), but got {surf.shape}"
        if surf.shape[1] == 3:
            surf = (surf + self.offset) / self.interval
            surf = surfaceutils.fill_grid(surf[:, :3], self.shape, self._interp, method, fill) # yapf: disable
        else:
            surf = (surf + self.offset[2]) / self.interval[2]
        assert surf.shape == self.shape, f"surf's shape {surf.shape} dosen't match the input shape {self.shape}"
        self.surf = surf

        self.mask = np.logical_or(surf < 0, np.isnan(surf))
        vertices, faces = surface2mesh(
            surf,
            self.mask,
            anti_rot=self.anti_rot,
            step1=self.steps[0],
            step2=self.steps[1],
        )
        return vertices, faces

    @property
    def offset(self):
        return self._offset

    @property
    def interval(self):
        return self._interval

    @property
    def cmaps(self):
        return self._cmaps

    @cmaps.setter
    def cmaps(self, cmaps):
        if self._cmaps == cmaps:
            return
        self._cmaps = cmaps
        self.process_values()

    @property
    def clims(self):
        return self._clims

    @clims.setter
    def clims(self, clims):
        if self._clims == clims:
            return
        self._clims = clims
        self.process_values()

    def update_offset_and_interval(self, offset, interval):
        update = False
        if offset is not None and np.any(np.array(offset) != self._offset):
            self._offset = np.array(offset)
            update = True
        if interval is not None and np.any(
                np.array(interval) != self._interval):
            self._interval = np.array(interval)
            update = True
        if update:
            vertices, faces = self.to_meshs()
            self._meshdata.set_vertices(vertices)
            self._meshdata.set_faces(faces)
            self.mesh_data_changed()

    def interp_value(self):
        if self.volume is None:
            raise RuntimeError('to interp, volume must be input')
        if self.seis_value is not None:
            return
        self.seis_value = surfaceutils.interp_surf(self.volume, self.surf)

    def process_values(self):
        if not isinstance(self.values, List):
            self.values = [self.values]
        if not isinstance(self._cmaps, List):
            self._cmaps = [self._cmaps]

        if self._clims is None:
            self._clims = [None] * len(self.values)
        if isinstance(self._clims, (List, Tuple)):
            if len(self._clims) > 1 and not isinstance(self._clims[0], (List, Tuple)): # yapf: disable
                self._clims = [self._clims]

        assert len(self._clims) == len(self.values), "clims must be the same length as values" # yapf: disable
        assert len(self._cmaps) == len(self.values), "cmaps must be the same length as values" # yapf: disable

        for i, value in enumerate(self.values):
            if isinstance(value, str) and value == 'depth':
                self.values[i] = self.surf
            elif isinstance(value, str) and value == 'amp':
                self.interp_value()
                self.values[i] = self.seis_value
                if self._clims[i] is None:
                    self._clims[i] = [
                        np.nanmin(self.volume),
                        np.nanmax(self.volume)
                    ]
            elif isinstance(value, str):
                try:
                    c = mcolors.to_rgb(value)
                    self.values[i] = c
                except:
                    raise ValueError(f"Invalid value {value}")
            elif isinstance(value, tuple) and isinstance(
                    value[0], str) and len(value) == 2:
                try:
                    assert value[1] <= 1 and value[
                        1] >= 0, "alpha must between 0 and 1"
                    c = mcolors.to_rgba(value[0], value[1])
                    self.values[i] = c
                except:
                    raise ValueError(f"Invalid value {value}")
            elif isinstance(value,
                            tuple) and len(value) >= 3 and len(value) <= 4:
                assert all([isinstance(v, (int, float)) for v in value]), "value must be a tuple of numbers" # yapf: disable
                assert all([v <= 1 and v >= 0 for v in value]), "value must be a tuple of numbers between 0 and 1" # yapf: disable
                self.values[i] = value
            elif not isinstance(value, np.ndarray):
                raise ValueError(
                    f"value must be 'depth', 'amp', or np.ndarray, but got {value}"
                )
            if self._clims[i] is None:
                self._clims[i] = [
                    np.nanmin(self.values[i]),
                    np.nanmax(self.values[i])
                ]
            self._cmaps[i] = colormap.get_cmap_from_str(self._cmaps[i])

        if len(self.values) == 1:
            if isinstance(self.values[0], tuple) and len(self.values[0]) <= 4:
                if self.render_type == 0 or self.render_type == -1:  # no change
                    self.color = self.values[0]
                else:
                    self.set_data(vertices=self._meshdata.get_vertices(),
                                  faces=self._meshdata.get_faces(),
                                  color=self.values[0])
                self.render_type = 0
            else:
                value = self.values[0][::self.steps[0], ::self.steps[1]].flatten() # yapf: disable
                mask = self.mask[::self.steps[0], ::self.steps[1]].flatten()
                value = value[~mask]
                if self.render_type == 1 or self.render_type == -1:
                    self._meshdata.set_vertex_values(value)
                else:
                    self.set_data(vertices=self._meshdata.get_vertices(),
                                  faces=self._meshdata.get_faces(),
                                  vertex_values=value)
                self.render_type = 1
                self.cmap = colormap.cmap_to_vispy(self._cmaps[0])
                self.clim = self._clims[0]
        else:
            colors = colormap.arrs_to_image(self.values, self._cmaps, self._clims) # yapf: disable
            colors = colors[::self.steps[0], ::self.steps[1]].reshape(-1, 4)
            mask = self.mask[::self.steps[0], ::self.steps[1]].flatten()
            colors = colors[~mask, :]
            if self.render_type == 2 or self.render_type == -1:
                self._meshdata.set_vertex_colors(colors)
            else:
                self.set_data(vertices=self._meshdata.get_vertices(),
                              faces=self._meshdata.get_faces(),
                              vertex_colors=colors)
            self.render_type = 2

    def update_colors_by_slice_node(self, nodes, volumes):
        node = [n for n in nodes if isinstance(n, AxisAlignedImage)]
        if len(node) == 0:
            raise ValueError("No AxisAlignedImage node found")
        if not isinstance(volumes, List):
            volumes = [volumes]

        node = node[0]
        cmaps = [img.cmap for img in node.overlaid_images]
        clims = [img.clim for img in node.overlaid_images]
        if len(cmaps) != len(volumes):
            raise ValueError("The number of volumes must be equal to the number of images in the AxisAlignedImage node") # yapf: disable

        values = [surfaceutils.interp_surf(volume, self.surf) for volume in volumes] # yapf: disable
        self._cmaps = cmaps
        self._clims = clims
        self.values = values
        self.process_values()


class ArbLineNode(Mesh):

    def __init__(
        self,
        path=None,
        anchor=None,
        data=None,
        volume=None,
        cmap='gray',
        clim=None,
        hstep=1,
        vstep=1,
        **kwargs,
    ):
        self.preprocess(path, anchor, data, volume)
        if clim is None:
            clim = [np.nanmin(self.data), np.nanmax(self.data)]

        self.nl, self.nt = self.data.shape
        assert len(self.path) == self.nl, "the length of path must be equal to the number of lines" # yapf: disable
        vertices, faces = arbline2mesh(self.path[::hstep],
                                       self.nt,
                                       False,
                                       vstep=vstep)
        values = self.data[::hstep, ::vstep].flatten()

        super().__init__(
            vertices=vertices,
            faces=faces,
            vertex_values=values,
            # don't set shading, as it will be lighted with gloss and shadow
            shading=None,
            **kwargs)
        self.cmap = colormap.cmap_to_vispy(cmap)
        self.clim = clim

    def preprocess(self, path=None, anchor=None, data=None, volume=None):
        if path is not None and anchor is not None:
            self.path = path
            self.anchor = None
            warnings.warn(
                "Both 'path' and 'anchor' are provided. Using 'path'.",
                category=UserWarning)
        elif path is not None:
            self.path = path
            self.anchor = None
        elif anchor is not None:
            self.path = None
            self.anchor = anchor
        else:
            raise ValueError("Either 'path' or 'anchor' must be provided.")

        # Check data and volume
        if data is not None and volume is not None:
            warnings.warn(
                "Both 'data' and 'volume' are provided. Using 'data'.",
                category=UserWarning)
        elif data is None and volume is None:
            raise ValueError("Either 'data' or 'volume' must be provided.")

        self.data = data
        self.volume = volume

        if self.path is None:
            self.path, _ = surfaceutils.interpolate_path(self.anchor)
        if self.data is None:
            pout, pdata = surfaceutils.extract_data(self.volume, self.path)
            self.data = surfaceutils.interp_arb(pout, pdata)
