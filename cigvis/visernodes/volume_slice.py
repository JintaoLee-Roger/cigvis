from typing import List, Dict, Tuple, Union
import time
from cigvis import colormap
import matplotlib.pyplot as plt
import viser

def arrs2img(bg, cmap, clim, masks, fg_cmaps, fg_clims):
    """
    merge serveral arrays into one image
    """
    if masks is not None and isinstance(masks, List) and len(masks) > 0:
        return colormap.blend_multiple(bg, masks, cmap, fg_cmaps, clim,
                                       fg_clims)
    else:
        norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
        out = colormap.get_cmap_from_str(cmap)(norm(bg))
        return out[:, :, :3]


class VolumeSlice:

    def __init__(self,
                 volume,
                 axis='z',
                 pos=0,
                 cmap='gray',
                 clim=None,
                 scale=-1):
        self._server: viser.ViserServer = None
        self.volume = volume
        self.axis = axis
        self.pos = pos
        self.cmap = cmap
        self.clim = clim if clim is not None else [volume.min(), volume.max()]
        self.scale = scale if scale > 0 else max(volume.shape) / 8
        self.update_nodes(self.pos)
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
        self.update_nodes(self.pos)

    @property
    def render_width(self):
        if self.axis != 'z':
            return self.volume.shape[2] / self.scale
        else:
            return self.volume.shape[1] / self.scale

    @property
    def render_height(self):
        if self.axis == 'x':
            return self.volume.shape[1] / self.scale
        else:
            return self.volume.shape[0] / self.scale

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
        if self.axis == 'x':
            ni = self.volume.shape[0]
            ri = ni / self.scale
            return (-ri / 2 + self.pos * ri / (ni - 1), 0, 0)
        elif self.axis == 'y':
            nx = self.volume.shape[1]
            rx = nx / self.scale
            return (0, -rx / 2 + self.pos * rx / (nx - 1), 0)
        else:
            nt = self.volume.shape[2]
            rt = nt / self.scale
            return (0, 0, -rt / 2 + self.pos * rt / (nt - 1))

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

        img = arrs2img(bg, self.cmap, self.clim, fg, self.fg_cmaps,
                       self.fg_clims)
        return img

    def update_nodes(self, pos):
        if self.server is None:
            return
        startt = time.perf_counter()
        self.pos = int(pos)
        self._check_bound()
        img = self.to_img()
        midt = time.perf_counter()
        self.nodes = self.server.scene.add_image(
            self.axis,
            img,
            self.render_width,
            self.render_height,
            'png',
            wxyz=self.wxyz,
            position=self.position,
        )
        endt = time.perf_counter()
        print(
            f"update_nodes:{(midt - startt)*1000:.3f}ms, {(endt - midt)*1000:.3f}ms"
        )

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
        self.update_nodes(self.pos)

    def update_clim(self, clim):
        self.clim = clim
        self.update_nodes(self.pos)

    def update_scale(self, scale):
        self.scale = scale
        self.update_nodes(self.pos)

    def add_mask(self, vol, cmap: str, clim: List = None):
        assert vol.shape == self.volume.shape
        self.masks.append(vol)
        if clim is None:
            clim = [vol.min(), vol.max()]
        self.fg_cmaps.append(cmap)
        self.fg_clims.append(clim)
        self.update_nodes(self.pos)

