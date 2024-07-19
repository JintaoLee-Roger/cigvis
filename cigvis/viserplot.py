import time

from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from cigvis import colormap
import numpy as np
import viser
from cigvis.visernodes import VolumeSlice


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  **kwargs) -> List:
    # set pos
    ni, nx, nt = volume.shape
    if pos is None:
        pos = dict(x=[0], y=[0], z=[nt - 1])
    if isinstance(pos, List):
        assert len(pos) == 3
        if isinstance(pos[0], List):
            x, y, z = pos
        else:
            x, y, z = [pos[0]], [pos[1]], [pos[2]]
        pos = {'x': x, 'y': y, 'z': z}
    assert isinstance(pos, Dict)

    if clim is None:
        clim = [volume.min(), volume.max()]

    nodes = []
    for axis, p in pos.items():
        for i in p:
            nodes.append(VolumeSlice(volume, axis, i, cmap, clim, **kwargs))

    return nodes


def add_mask(nodes: List,
             volumes: Union[List[np.ndarray], np.ndarray],
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, List] = None,
             **kwargs) -> List:
    if not isinstance(volumes, List):
        volumes = [volumes]

    # for volume in volumes:
    #     # TODO: check shape as same as base image
    #     utils.check_mmap(volume)

    if clims is None:
        clims = [[v.min(), v.max()] for v in volumes]
    if not isinstance(clims[0], (List, Tuple)):
        clims = [clims]

    if cmaps is None:
        raise ValueError("'cmaps' cannot be 'None'")
    if not isinstance(cmaps, List):
        cmaps = [cmaps] * len(volumes)
    for i in range(len(cmaps)):
        cmaps[i] = colormap.get_cmap_from_str(cmaps[i])

    for node in nodes:
        if not isinstance(node, VolumeSlice):
            continue
        for i in range(len(volumes)):
            node.add_mask(
                volumes[i],
                cmaps[i],
                clims[i],
            )

    return nodes


def plot3D(nodes):
    server = viser.ViserServer()

    for node in nodes:
        node.server = server

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.fov = -5

    server.scene.set_up_direction((0.0, 0.0, -1.0))

    with server.gui.add_folder("slices pos"):
        nodex = [node for node in nodes if node.axis == 'x'][0]
        nodey = [node for node in nodes if node.axis == 'y'][0]
        nodez = [node for node in nodes if node.axis == 'z'][0]
        guix = server.gui.add_slider(
            'x',
            min=0,
            max=nodex.limit[1] - 1,
            step=1,
            initial_value=nodex.pos,
        )
        guix.on_update(lambda _: nodex.update_nodes(guix.value))
        guiy = server.gui.add_slider(
            'y',
            min=0,
            max=nodey.limit[1] - 1,
            step=1,
            initial_value=nodey.pos,
        )
        guiy.on_update(lambda _: nodey.update_nodes(guiy.value))
        guiz = server.gui.add_slider(
            'z',
            min=0,
            max=nodez.limit[1] - 1,
            step=1,
            initial_value=nodez.pos,
        )
        guiz.on_update(lambda _: nodez.update_nodes(guiz.value))

    vmin = nodes[0].volume.min()
    vmax = nodes[0].volume.max()
    step = (vmax - vmin) / 100

    def update_clim(vmin, vmax):
        if vmin >= vmax:
            return
        for node in nodes:
            node.update_clim([vmin, vmax])

    def update_cmap(cmap):
        for node in nodes:
            node.update_cmap(cmap)

    with server.gui.add_folder("paramters"):
        guivmin = server.gui.add_slider(
            'vmin',
            min=vmin,
            max=vmax,
            step=step,
            initial_value=nodes[0].clim[0],
        )

        guivmax = server.gui.add_slider(
            'vmax',
            min=vmin,
            max=vmax,
            step=step,
            initial_value=nodes[0].clim[1],
        )

        guivmin.on_update(lambda _: update_clim(guivmin.value, guivmax.value))
        guivmax.on_update(lambda _: update_clim(guivmin.value, guivmax.value))

        guicmap = server.gui.add_dropdown(
            'cmap',
            options=[
                'gray', 'seismic', 'Petrel', 'stratum', 'jet', 'od_seismic1',
                'od_seismic2', 'od_seismic3'
            ],
            initial_value='gray',
        )
        guicmap.on_update(lambda _: update_cmap(guicmap.value))

    while True:
        time.sleep(0.1)
