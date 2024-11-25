from typing import List, Dict, Tuple, Union
import time
import numpy as np
from cigvis import colormap, ExceptionWrapper
import matplotlib.pyplot as plt
from cigvis.utils import utils

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
        self.clim = clim if clim is not None else utils.auto_clim()

        self.init_scale = [1.5 / max(volume.shape)] * 3
        self.nancolor = nancolor

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
            return self.volume.shape[2] * self.scale[2]
        else:
            return self.volume.shape[1] * self.scale[1]

    @property
    def render_height(self):
        if self.axis == 'x':
            return self.volume.shape[1] * self.scale[1]
        else:
            return self.volume.shape[0] * self.scale[0]

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
        ni, nx, nt = self.volume.shape
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

    def to_img(self):
        if self.axis == 'x':
            bg = self.volume[self.pos, :, :]
            fg = [mask[self.pos, :, :] for mask in self.masks]
        elif self.axis == 'y':
            bg = self.volume[:, self.pos, :]
            fg = [mask[:, self.pos, :] for mask in self.masks]
        else:
            bg = self.volume[:, :, self.pos]
            fg = [mask[:, :, self.pos] for mask in self.masks]

        bg = self._to_np(bg)
        fg = [self._to_np(g) for g in fg]

        img = colormap.arrs_to_image([bg] + fg, [self.cmap] + self.fg_cmaps,
                                     [self.clim] + self.fg_clims, True,
                                     self.nancolor)
        return img

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
            'png',
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
            return (0, self.volume.shape[0])
        elif self.axis == 'y':
            return (0, self.volume.shape[1])
        else:
            return (0, self.volume.shape[2])

    def _check_bound(self):
        if self.pos < 0:
            self.pos = 0
        if self.pos >= self.limit[1]:
            self.pos = self.limit[1] - 1

    def update_cmap(self, cmap):
        self.cmap = cmap
        self.update_node(self.pos)

    def update_clim(self, clim):
        self.clim = clim
        self.update_node(self.pos)

    def update_scale(self, scale):
        if isinstance(scale, (int, float)):
            scale = [scale] * 3
        self.scale = [s * x for s, x in zip(scale, self.init_scale)]

        self.update_node(self.pos)

    def add_mask(self, vol, cmap: str, clim: List = None):
        assert vol.shape == self.volume.shape
        self.masks.append(vol)
        if clim is None:
            clim = utils.auto_clim(vol)
        self.fg_cmaps.append(cmap)
        self.fg_clims.append(clim)
        self.update_node(self.pos)
