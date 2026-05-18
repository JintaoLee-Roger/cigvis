from typing import Optional
import warnings
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

from cigvis import colormap, ExceptionWrapper
from cigvis.meshs.surfaces import arbline2mesh
from cigvis.utils import surfaceutils
from .base import ViserNodeMixin, auto_scale_from_points, color_to_uint8_rgb

try:
    import viser
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals
    from PIL import Image
except BaseException as E:
    message = "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to enable viser"
    viser = ExceptionWrapper(E, message)
    trimesh = ExceptionWrapper(E, message)
    PBRMaterial = ExceptionWrapper(E, message)
    TextureVisuals = ExceptionWrapper(E, message)
    Image = ExceptionWrapper(E, message)


def color_f2i(colors: ArrayLike):
    colors = np.clip(colors, 0, 1) * 255
    return colors.astype(np.uint8)


def color_i2f(colors: ArrayLike):
    colors = np.clip(colors.astype(np.float32), 0, 255) / 255
    return colors


def color2textual(colors: ArrayLike, vertices: ArrayLike):
    x_indices = np.round(vertices[:, 0]).astype(int)
    y_indices = np.round(vertices[:, 1]).astype(int)

    max_x = x_indices.max() + 1
    max_y = y_indices.max() + 1

    img_array = np.zeros((max_y, max_x, 4), dtype=np.uint8)

    img_array[y_indices, x_indices] = colors
    img_array = img_array[::-1, :, :]
    img_pil = Image.fromarray(img_array)

    uv = np.column_stack((x_indices / (max_x - 1), y_indices / (max_y - 1)))

    return img_pil, uv


class MeshNode(ViserNodeMixin, trimesh.Trimesh):
    """
    Note: All colors should be in the range [0, 255] in uint8 format.
    """

    def __init__(
        self,
        vertices: Optional[ArrayLike] = None,
        faces: Optional[ArrayLike] = None,
        face_colors: Optional[ArrayLike] = None,
        vertex_colors: Optional[ArrayLike] = None,
        color=(90, 200, 255),
        vertices_values: Optional[ArrayLike] = None,
        scale=-1,
        cmap: Optional[str] = 'jet',
        clim: Optional[ArrayLike] = None,
        **kwargs,
    ):
        trimesh.Trimesh.__init__(self, vertices=vertices, faces=faces, **kwargs)

        self._base_vertices = np.asarray(vertices, dtype=np.float32)
        self._vertices = self._base_vertices
        self.colored_by = None  # can be 'value', 'vertex', 'face', or 'uniform'

        if scale < 0:
            scale = auto_scale_from_points(self._base_vertices, target=1.0)

        self._cmap = cmap
        self._clim = clim
        self._face_colors = face_colors
        self._vertex_colors = vertex_colors
        self._vertices_values = vertices_values
        self._color = color_to_uint8_rgb(color)
        self._set_color = False
        self._init_node_state('mesh', scale)
        self._on_scale_changed()

        if self._face_colors is not None:
            self.colored_by = 'face'

        if self.colored_by is None and self._vertex_colors is not None:
            self.colored_by = 'vertex'

        if self.colored_by is None and self._vertices_values is not None:
            self.colored_by = 'value'

        if self.colored_by is None:
            self.colored_by = 'uniform'

    def set_colors(self):
        if self.server is None:
            return
        if self._set_color:
            return

        self._set_color = True
        if self.colored_by == 'value':
            if self.clim is None:
                self._clim = [
                    self._vertices_values.min(),
                    self._vertices_values.max()
                ]
            norm = plt.Normalize(vmin=self.clim[0], vmax=self.clim[1])
            colors = colormap.get_cmap_from_str(self._cmap)(norm(
                self._vertices_values))
            self.visual.vertex_colors = color_f2i(colors)
            return

        if self.colored_by == 'vertex':
            self.visual.vertex_colors = self._vertex_colors
        elif self.colored_by == 'face':
            self.visual.face_colors = self._face_colors
        elif self.colored_by == 'uniform':
            self.visual.vertex_colors = np.tile(
                self._color,
                (self.vertices.shape[0], 1),
            )
        self._set_visual()

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap
        self._set_color = False
        self.update_node()

    @property
    def clim(self):
        return self._clim

    @clim.setter
    def clim(self, clim):
        self._clim = clim
        self._set_color = False
        self.update_node()

    @property
    def wxyz(self):
        pass

    @property
    def position(self):
        pass

    @property
    def data_extent(self):
        return np.ptp(self._base_vertices[:, :3], axis=0)

    def _on_scale_changed(self):
        self.vertices = self._base_vertices * np.asarray(self.scale)

    def _set_visual(self):
        self.visual = self.visual.to_texture()
        self.visual.material = self.visual.material.to_pbr()
        self.visual.material.doubleSided = True
        self.visual.material.roughnessFactor = 0.72
        self.visual.material.metallicFactor = 0.0
        self.visual.material.baseColorFactor = [1.0, 1.0, 1.0, 1.0]

    def update_node(self):
        if self.server is None:
            return

        self.set_colors()

        # trimesh's GLB exporter identifies meshes by class name through its
        # internal MRO helper. With our multiple-inheritance node class, recent
        # trimesh versions may export an empty GLB unless we pass a plain mesh.
        mesh = self.copy()
        self.nodes = self.server.scene.add_mesh_trimesh(
            self.name,
            mesh,
        )


class SurfaceNode(MeshNode):

    def __init__(
        self,
        vertices: Optional[ArrayLike] = None,
        faces: Optional[ArrayLike] = None,
        face_colors: Optional[ArrayLike] = None,
        vertex_colors: Optional[ArrayLike] = None,
        color=(90, 200, 255),
        vertices_values: Optional[ArrayLike] = None,
        scale: Optional[float] = 1.0,
        cmap: Optional[str] = 'jet',
        clim: Optional[ArrayLike] = None,
        **kwargs,
    ):
        super().__init__(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,
            vertex_colors=vertex_colors,
            color=color,
            vertices_values=vertices_values,
            scale=scale,
            cmap=cmap,
            clim=clim,
            **kwargs,
        )

class ArbLineNode(MeshNode):

    def __init__(
        self,
        path=None,
        anchor=None,
        data=None,
        volume=None,
        scale=-1,
        cmap: Optional[str] = 'jet',
        clim: Optional[ArrayLike] = None,
        hstep=1,
        vstep=1,
        **kwargs,
    ):
        self.hstep = hstep 
        self.vstep = vstep
        self.preprocess(path, anchor, data, volume)
        if clim is None:
            clim = [np.nanmin(self.data), np.nanmax(self.data)]

        self.nl, self.nt = self.data.shape
        assert len(self.path) == self.nl

        vertices, faces = arbline2mesh(self.path[::hstep], self.nt, False, vstep=vstep)
        super().__init__(
            vertices=vertices,
            faces=faces,
            scale=scale,
            cmap=cmap,
            clim=clim,
            **kwargs,
        )
        self.data = data

    def set_colors(self):
        if self.server is None:
            return
        if self._set_color:
            return

        self._set_color = True
        if self.clim is None:
            self.clim = [np.nanmin(self.data), np.nanmax(self.data)]

        norm = plt.Normalize(vmin=self.clim[0], vmax=self.clim[1])
        colors = colormap.get_cmap_from_str(self._cmap)(norm(self.data[::self.hstep, ::self.vstep]))
        colors = color_f2i(colors)
        self.visual.vertex_colors = colors.reshape(-1, 4)
        self._set_visual()

    def _set_visual(self):
        self.visual = self.visual.to_texture()
        self.visual.material = self.visual.material.to_pbr()
        self.visual.material.doubleSided = True
        self.visual.material.roughnessFactor = 0.72
        self.visual.material.metallicFactor = 0.0
        self.visual.material.baseColorFactor = [255, 255, 255, 255]


    def preprocess(self, path=None, anchor=None, data=None, volume=None):
        if path is not None and anchor is not None:
            self.path = path
            self.anchor = None
            warnings.warn(
                "Both 'path' and 'anchor' are provided. Using 'path'.",
                category=UserWarning
            )
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
                category=UserWarning
            )
        elif data is None and volume is None:
            raise ValueError("Either 'data' or 'volume' must be provided.")

        self.data = data 
        self.volume = volume

        if self.path is None:
            self.path, _ = surfaceutils.interpolate_path(self.anchor)
        if self.data is None:
            pout, pdata = surfaceutils.extract_data(self.volume, self.path)
            self.data = surfaceutils.interp_arb(pout, pdata)
        
