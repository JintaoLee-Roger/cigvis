from typing import Optional
import warnings
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt

from cigvis import colormap, ExceptionWrapper
from cigvis.meshs.surfaces import arbline2mesh
from cigvis.utils import surfaceutils

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


class MeshNode(trimesh.Trimesh):
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
        super().__init__(vertices=vertices, faces=faces, **kwargs)

        self._vertices = vertices
        self.colored_by = None  # can be 'value', 'vertex', 'face', or 'uniform'

        if scale < 0:
            rx = vertices[:, 0].max() - vertices[:, 0].min()
            ry = vertices[:, 1].max() - vertices[:, 1].min()
            rz = vertices[:, 2].max() - vertices[:, 2].min()
            self._scale = [1 / max([rx, ry, rz])] * 3
        else:
            self._scale = [scale] * 3

        self._cmap = cmap
        self._clim = clim
        self._server = None # viser.ViserServer
        self._face_colors = face_colors
        self._vertex_colors = vertex_colors
        self._vertices_values = vertices_values
        self._color = color
        self._name = 'mesh'
        self._set_color = False

        if self._face_colors is not None:
            self.colored_by = 'face'

        if self.colored_by is None and self._vertex_colors is not None:
            self.colored_by = 'vertex'

        if self.colored_by is None and self._vertices_values is not None:
            self.colored_by = 'value'

        if self.colored_by is None:
            self.colored_by = 'uniform'

    def set_colors(self):
        pass

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, server):
        if not isinstance(server, viser.ViserServer):
            raise ValueError("server must be type: viser.ViserServer")
        self._server = server
        self._set_color = False
        self.update_node()

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
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        self.update_node()

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
        self.vertices[:, 0] = self._vertices[:, 0] * self._scale[0]
        self.vertices[:, 1] = self._vertices[:, 1] * self._scale[1]
        self.vertices[:, 2] = self._vertices[:, 2] * self._scale[2]
        self.update_node()

    def update_node(self):
        if self.server is None:
            return

        self.set_colors()

        self.nodes = self.server.scene.add_mesh_trimesh(
            self.name,
            self,
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

    def set_colors(self):
        if self.server is None:
            return
        if self._set_color:
            return

        self._set_color = True
        if self.colored_by == 'value':
            if self.clim is None:
                self.clim = [
                    self._vertices_values.min(),
                    self._vertices_values.max()
                ]
            norm = plt.Normalize(vmin=self.clim[0], vmax=self.clim[1])
            colors = colormap.get_cmap_from_str(self._cmap)(norm(
                self._vertices_values))
            colors = color_f2i(colors)
            # self.visual.vertex_colors = colors
            img, uv = color2textual(colors, self._vertices)
            self.visual = TextureVisuals(
                uv,
                PBRMaterial(
                    roughnessFactor=0.4,
                    baseColorFactor=[110, 110, 110, 255],
                    metallicFactor=.4,
                    baseColorTexture=img,
                    doubleSided=True,
                ))
            return

        elif self.colored_by == 'vertex':
            self.visual.vertex_colors = self._vertex_colors
        elif self.colored_by == 'face':
            self.visual.face_colors = self._face_colors
        elif self.colored_by == 'uniform':
            self.visual.vertex_colors = np.tile(
                self._color,
                (self.vertices.shape[0], 1),
            )
        self._set_visual()

    def _set_visual(self):
        self.visual = self.visual.to_texture()
        self.visual.material = self.visual.material.to_pbr()
        self.visual.material.doubleSided = True
        self.visual.material.roughnessFactor = 0.4
        self.visual.material.metallicFactor = 0.4
        self.visual.material.baseColorFactor = [110, 110, 110, 255]


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

        self.nl, self.nt = data.shape
        assert len(path) == self.nl

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
        self.visual.material.roughnessFactor = 0.4
        self.visual.material.metallicFactor = 0.2
        self.visual.material.baseColorFactor = [100, 100, 100, 255]


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
        