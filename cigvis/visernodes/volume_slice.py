from typing import List, Dict, Tuple, Union
import time
import numpy as np
from cigvis import colormap, ExceptionWrapper
import matplotlib.pyplot as plt
from cigvis.utils import utils
from cigvis import is_line_first

try:
    import viser
except BaseException as E:
    message = "run `pip install \"cigvis[viser]\"` or run `pip install \"cigvis[all]\"` to enable viser"
    viser = ExceptionWrapper(E, message)


class VolumeSlice:

    def __init__(self,
                 volume,
                 axis='z',
                 pos=0,
                 cmap='gray',
                 clim=None,
                 scale=-1,
                 nancolor=None):
        self._server = None  # viser.ViserServer
        self.volume = volume
        self.axis = axis
        self.pos = pos
        self.cmap = cmap
        self._cmap_preset = cmap
        self.clim = clim if clim is not None else utils.auto_clim(volume)

        self.init_scale = [1.5 / max(volume.shape)] * 3
        self.nancolor = nancolor

        # HACK: to deal with rgb or rgba image
        def _eq_3_or_4(k):
            return k == 3 or k == 4
        line_first = is_line_first()
        assert _eq_3_or_4(volume.ndim), f"Volume's dims must be 3 or 4 (RGB), but got {volume.ndim}"
        # rgb_type, 0 for (n1, n2, n3), 1 for (n1, n2, n3, 3/4), 2 for (3/4, n1, n2, n3)
        ndim = volume.ndim
        self.channel_dim = None
        dim_x, dim_y, dim_z = (0, 1, 2) if line_first else (2, 1, 0)
        self.axis_to_dim = {'x': dim_x, 'y': dim_y, 'z': dim_z}
        self.vol_shape, self.rgb_type = utils.get_shape(volume, line_first)
        if self.rgb_type == 1:
            self.channel_dim = 3
        elif self.rgb_type == 2:
            self.channel_dim = 0

        if isinstance(scale, (int, float)):
            if scale < 0:
                self.scale = [1] * 3
            else:
                self.scale = [scale] * 3
        else:
            self.scale = scale
        self.scale = [s * x for s, x in zip(self.scale, self.init_scale)]

        self.update_node(self.pos)
        self.masks = []
        self.fg_cmaps = []
        self.fg_clims = []
        self._fg_cmaps_preset = []

    @property
    def server(self):
        return self._server

    @server.setter
    def server(self, server):
        if not isinstance(server, viser.ViserServer):
            raise ValueError("server must be type: viser.ViserServer")
        self._server = server
        self.update_node(self.pos)

    @property
    def render_width(self):
        if self.axis != 'z':
            return self.vol_shape[2] * self.scale[2]
        else:
            return self.vol_shape[1] * self.scale[1]

    @property
    def render_height(self):
        if self.axis == 'x':
            return self.vol_shape[1] * self.scale[1]
        else:
            return self.vol_shape[0] * self.scale[0]

    @property
    def wxyz(self):
        if self.axis == 'x':
            return (0.7071, 0.0, -0.7071, 0.0)
        elif self.axis == 'y':
            return (-0.5, 0.5, 0.5, 0.5)
        else:
            return (0, 0.7071, 0.7071, 0)

    @property
    def position(self):
        ni, nx, nt = self.vol_shape
        ri = ni * self.scale[0]
        rx = nx * self.scale[1]
        rt = nt * self.scale[2]
        if self.axis == 'x':
            return (self.pos * ri / (ni - 1), rx / 2, rt / 2)
        elif self.axis == 'y':
            return (ri / 2, self.pos * rx / (nx - 1), rt / 2)
        else:
            return (ri / 2, rx / 2, self.pos * rt / (nt - 1))

    def _to_np(self, img):
        if utils.is_torch_tensor(img):
            img = img.detach().cpu().numpy()
        return img

    def _get_slices(self, axis, pos):
        dim = self.axis_to_dim.get(axis)
        slices = [slice(None)] * self.volume.ndim
        if self.channel_dim is not None and dim >= self.channel_dim:
            slices[dim + 1] = pos
        else:
            slices[dim] = pos
        return tuple(slices)

    def _auto_transpose(self, img):
        if not is_line_first():
            img = np.transpose(img, (1, 0, 2))
        return img

    def to_img(self):
        s = self._get_slices(self.axis, self.pos)
        bg = self.volume[s]
        fg = [mask[s] for mask in self.masks]

        bg = self._to_np(bg)
        fg = [self._to_np(g) for g in fg]

        img = colormap.arrs_to_image([bg] + fg, [self.cmap] + self.fg_cmaps,
                                     [self.clim] + self.fg_clims, True,
                                     self.nancolor)
        return self._auto_transpose(img)

    def update_node(self, pos):
        if self.server is None:
            return
        # startt = time.perf_counter()
        self.pos = int(pos)
        self._check_bound()
        img = self.to_img()
        # midt = time.perf_counter()
        self.nodes = self.server.scene.add_image(
            self.axis,
            img,
            self.render_width,
            self.render_height,
            format='png',
            wxyz=self.wxyz,
            position=self.position,
        )
        # endt = time.perf_counter()
        # print(
        #     f"update_node:{(midt - startt)*1000:.3f}ms, {(endt - midt)*1000:.3f}ms"
        # )

    @property
    def limit(self):
        if self.axis == 'x':
            return (0, self.vol_shape[0])
        elif self.axis == 'y':
            return (0, self.vol_shape[1])
        else:
            return (0, self.vol_shape[2])

    def _check_bound(self):
        if self.pos < 0:
            self.pos = 0
        if self.pos >= self.limit[1]:
            self.pos = self.limit[1] - 1

    def update_cmap(self, cmap):
        if cmap is None:
            cmap = self._cmap_preset
        self.cmap = cmap
        self.update_node(self.pos)

    def update_clim(self, clim):
        self.clim = clim
        self.update_node(self.pos)

    def update_mask_clim(self, clim, num):
        if len(self.masks) == 0 or num >= len(self.masks):
            return 
        self.fg_clims[num] = clim 
        self.update_node(self.pos)

    def update_mask_cmap(self, cmap, num):
        if len(self.masks) == 0 or num >= len(self.masks):
            return 
        self.fg_cmaps[num] = cmap 
        self.update_node(self.pos)

    def update_scale(self, scale):
        if isinstance(scale, (int, float)):
            scale = [scale] * 3
        self.scale = [s * x for s, x in zip(scale, self.init_scale)]

        self.update_node(self.pos)

    def add_mask(self, vol, cmap: str, clim: List = None):
        mask_shape, _ = utils.get_shape(vol, is_line_first())
        assert mask_shape == self.vol_shape, f"mask.shape: {vol.shape} != vol.shape: {self.vol_shape}"
        self.masks.append(vol)
        if clim is None:
            clim = utils.auto_clim(vol)
        self.fg_cmaps.append(cmap)
        self._fg_cmaps_preset.append(cmap)
        self.fg_clims.append(clim)
        self.update_node(self.pos)
