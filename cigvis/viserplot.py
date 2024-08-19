import time

from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import viser

import cigvis
from cigvis import colormap
from cigvis.visernodes import (
    VolumeSlice,
    SurfaceNode,
    MeshNode,
)
from cigvis.meshs import surface2mesh
import cigvis.utils as utils
from cigvis.utils import surfaceutils


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
        clim = [utils.nmin(volume), utils.nmax(volume)]

    nodes = []
    for axis, p in pos.items():
        for i in p:
            nodes.append(VolumeSlice(volume, axis, i, cmap, clim, **kwargs))

    return nodes


def add_mask(nodes: List,
             volumes: Union[List, np.ndarray],
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, List] = None,
             **kwargs) -> List:
    if not isinstance(volumes, List):
        volumes = [volumes]

    for volume in volumes:
        # TODO: check shape as same as base image
        utils.check_mmap(volume)

    if clims is None:
        clims = [[utils.nmin(v), utils.nmax(v)] for v in volumes]
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


def create_surfaces(surfs: List[np.ndarray],
                    volume: np.ndarray = None,
                    value_type: str = 'depth',
                    clim: List = None,
                    cmap: str = 'jet',
                    alpha: float = 1,
                    shape: Union[Tuple, List] = None,
                    interp: bool = False,
                    step1=1,
                    step2=1,
                    **kwargs) -> List:
    """
    create a surfaces node

    Parameters
    ----------
    surfs : List or array-like
        the surface position, which can be an array (one surface) or 
        List (multi-surfaces). Each surf can be a (n1, n2)/(n1, n2, 2) 
        array or (N, 3)/(N, 4) array, such as
        >>> surf.shape = (n1, n2) # surf[i, j] means z pos at x=i, y=j
        # surf[i, j, 0] means z pos at x=i, y=j
        # surf[i, j, 1] means value for plotting at pos (i, j, surf[i, j])
        >>> surf.shape = (n1, n2, 2)
        # surf[i, j, 1:] means rgb or rgba color at pos (i, j, surf[i, j])
        >>> surf.shape = (n1, n2, 4) or (n1, n2, 5)
        >>> surf.shape = (N, 3) # surf[i, :] means i-th point position
        # surf[i, :3] means i-th point position
        # surf[i, 3] means i-th point's value for plotting
        >>> surf.shape = (N, 4)
        # surf[i, 3:] means i-th point color in rgb or rgba format
        >>> surf.shape = (N, 6) or (N, 7)
    volume : array-like
        3D array, values when surf_color is 'amp'
    value_type : str
        'depth' or 'amp', show z or amplitude, amplitude can be values in volume or
        values or colors
    clim : List
        [vmin, vmax] of surface volumes
    cmap : str or Colormap
        cmap for surface
    alpha : float
        opactity of the surfaces
    shape : List or Tuple
        If surf's shape is like (N, 3) or (N, 4), shape must be specified,
        if surf's shape is like (n1, n2) or (n1, n2, 2), shape will be ignored
    
    kwargs : Dict
        parameters for vispy.scene.visuals.Mesh
    """
    utils.check_mmap(volume)
    utils.check_mmap(surfs)
    line_first = cigvis.is_line_first()
    method = kwargs.get('method', 'cubic')
    fill = kwargs.get('fill', -1)
    anti_rot = kwargs.get('anti_rot', True)

    # add surface
    if not isinstance(surfs, List):
        surfs = [surfs]

    surfaces = []
    values = []
    colors = []
    for surf in surfs:
        if surf.ndim == 3:
            s, v, c = surfaceutils.preproc_surf_array3(surf, value_type)
        elif surf.ndim == 2:
            if surf.shape[1] > 7:
                s, v, c = surfaceutils.preproc_surf_array2(
                    surf, volume, value_type)
            else:
                assert volume is not None or shape is not None
                if shape is None:
                    shape = volume.shape[:2] if line_first else volume.shape[1:]
                s, v, c = surfaceutils.preproc_surf_pos(
                    surf, shape, volume, value_type, interp, method, fill)
        else:
            raise RuntimeError('Invalid shape')
        surfaces.append(s)
        values.append(v)
        colors.append(c)

    if value_type == 'depth':
        values = surfaces

    if clim is None and value_type == 'amp':
        vmin = min([utils.nmin(s) for s in values])
        vmax = max([utils.nmax(s) for s in values])
        clim = [vmin, vmax]
    elif clim is None and value_type == 'depth':
        vmin = min([s[s >= 0].min() for s in values])
        vmax = max([s[s >= 0].max() for s in values])
        clim = [vmin, vmax]

    cmap = colormap.get_cmap_from_str(cmap)
    if alpha < 1:
        cmap = colormap.set_alpha(cmap, alpha, False)

    mesh_nodes = []
    for s, v, c in zip(surfaces, values, colors):
        mask = np.logical_or(s < 0, np.isnan(s))
        vertices, faces = surface2mesh(
            s,
            mask,
            anti_rot=anti_rot,
            step1=step1,
            step2=step2,
        )
        mask = mask[::step1, ::step2]
        if v is not None:
            v = v[::step1, ::step2]
            v = v[~mask].flatten()
        if c is not None:
            channel = c.shape[-1]
            c = c[::step1, ::step2, ...]
            c = c[~mask].flatten().reshape(-1, channel)

        mesh_kwargs = {}  # TODO:

        if kwargs.get('color', None) is not None:
            v = None
            c = None
        if c is not None:
            v = None

        mesh = SurfaceNode(vertices=vertices,
                           faces=faces,
                           face_colors=None,
                           vertex_colors=c,
                           vertices_values=v,
                           **mesh_kwargs)

        if v is not None and c is None and kwargs.get('color', None) is None:
            mesh.cmap = cmap
            mesh.clim = clim

        mesh_nodes.append(mesh)

    return mesh_nodes


def plot3D(nodes, axis_scales=[1, 1, 1], **kwargs):
    server = viser.ViserServer(label='cigvis-viser')

    # update scale of slices
    draw_slices = False
    init_scale = -1
    for node in nodes:
        if isinstance(node, VolumeSlice):
            init_scale = node.init_scale
            node.update_scale(axis_scales)
            draw_slices = True

    if init_scale == -1:  # no slices # TODO: for other types, Well logs?
        init_scale = 100
        for node in nodes:
            if isinstance(node, MeshNode):
                init_scale = min(min(node.scale), init_scale)
        init_scale = [init_scale] * 3

    # update scale of meshes
    meshid = 0
    for node in nodes:
        if isinstance(node, MeshNode):
            node.scale = [s * x for s, x in zip(init_scale, axis_scales)]
            node.name = f'mesh{meshid}'
            meshid += 1
        node.server = server

    # gui slices slibers to control slices position
    with server.gui.add_folder("slices pos"):
        nodex = [
            node for node in nodes
            if isinstance(node, VolumeSlice) and node.axis == 'x'
        ]
        nodey = [
            node for node in nodes
            if isinstance(node, VolumeSlice) and node.axis == 'y'
        ]
        nodez = [
            node for node in nodes
            if isinstance(node, VolumeSlice) and node.axis == 'z'
        ]
        if len(nodex) > 0:
            nodex = nodex[0]
            guix = server.gui.add_slider(
                'x',
                min=0,
                max=nodex.limit[1] - 1,
                step=1,
                initial_value=nodex.pos,
            )
            guix.on_update(lambda _: nodex.update_node(guix.value))

        if len(nodey) > 0:
            nodey = nodey[0]
            guiy = server.gui.add_slider(
                'y',
                min=0,
                max=nodey.limit[1] - 1,
                step=1,
                initial_value=nodey.pos,
            )
            guiy.on_update(lambda _: nodey.update_node(guiy.value))

        if len(nodez) > 0:
            nodez = nodez[0]
            guiz = server.gui.add_slider(
                'z',
                min=0,
                max=nodez.limit[1] - 1,
                step=1,
                initial_value=nodez.pos,
            )
            guiz.on_update(lambda _: nodez.update_node(guiz.value))

    # gui to control slices clim and cmap
    if draw_slices:
        vmin = utils.nmin(nodes[0].volume)
        vmax = utils.nmax(nodes[0].volume)
        step = (vmax - vmin) / 100

    def update_clim(vmin, vmax):
        if vmin >= vmax:
            return
        for node in nodes:
            if hasattr(node, 'update_clim'):
                node.update_clim([vmin, vmax])

    def update_cmap(cmap):
        for node in nodes:
            if hasattr(node, 'update_cmap'):
                node.update_cmap(cmap)

    if draw_slices:
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

            guivmin.on_update(
                lambda _: update_clim(guivmin.value, guivmax.value))
            guivmax.on_update(
                lambda _: update_clim(guivmin.value, guivmax.value))

            guicmap = server.gui.add_dropdown(
                'cmap',
                options=[
                    'gray', 'seismic', 'Petrel', 'stratum', 'jet',
                    'od_seismic1', 'od_seismic2', 'od_seismic3'
                ],
                initial_value='gray',
            )
            guicmap.on_update(lambda _: update_cmap(guicmap.value))

    # gui to control aspect
    def update_scale(scale):
        for node in nodes:
            if isinstance(node, VolumeSlice):
                node.update_scale(scale)
            elif isinstance(node, MeshNode):
                node.scale = [s * x for s, x in zip(init_scale, scale)]

    with server.gui.add_folder('Aspect'):
        gui_scalex = server.gui.add_slider(
            'scale_x',
            min=0.25,
            max=2.5,
            step=0.25,
            initial_value=1,
        )

        gui_scaley = server.gui.add_slider(
            'scale_y',
            min=0.25,
            max=2.5,
            step=0.25,
            initial_value=1,
        )

        gui_scalez = server.gui.add_slider(
            'scale_z',
            min=0.1,
            max=3,
            step=0.1,
            initial_value=1,
        )

        gui_scalex.on_update(lambda _: update_scale(
            [gui_scalex.value, gui_scaley.value, gui_scalez.value]))
        gui_scaley.on_update(lambda _: update_scale(
            [gui_scalex.value, gui_scaley.value, gui_scalez.value]))
        gui_scalez.on_update(lambda _: update_scale(
            [gui_scalex.value, gui_scaley.value, gui_scalez.value]))

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.fov = np.pi / 4.0  # Or some other angle in radians

        # TODO: move the origin (0, 0, 0) to the top
        # def _(camera):
        client.camera.position = (8, 8, 0)
        # client.camera.wxyz = (0.7, 0, 0,0.7)

    server.scene.set_up_direction((0.0, 0.0, -1.0))


    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # ca = server.get_clients()
        # for k, cc in ca.items():
        #     c = cc.camera
        #     print(k, c.position, c.wxyz, c.fov, c.aspect, c.look_at, c.up_direction)
        server.stop()
        del server
        print("Execution interrupted")
