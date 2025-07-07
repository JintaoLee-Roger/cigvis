import time

from typing import List, Dict, Tuple, Union
import re
import matplotlib.pyplot as plt
import numpy as np
import viser
from PIL import Image, ImageDraw
import imageio.v3 as iio
from packaging import version


import cigvis
from cigvis import colormap
from cigvis.visernodes import (
    VolumeSlice,
    SurfaceNode,
    MeshNode,
    LogPoints,
    LogLineSegments,
    LogBase,
)
from cigvis.meshs import surface2mesh
import cigvis.utils as utils
from cigvis.utils import surfaceutils


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  nancolor=None,
                  **kwargs) -> List:
    """
    create a slice node

    Parameters
    ----------
    volume : array-like
        3D array
    pos : List or Dict
        init position of the slices, can be a List or Dict, such as:
        ```
        pos = [0, 0, 200] # x: 0, y: 0, z: 200
        pos = [[0, 200], [9], []] # x: 0 and 200, y: 9, z: None
        pos = {'x': [0, 200], 'y': [1], z: []}
        ```
    clim : List
        [vmin, vmax] for plotting 
    cmap : str or Colormap
        colormap, it can be str or matplotlib's Colormap
    nancolor : str or color
        color for nan values, default is None (i.e., transparent)
    """
    # set pos
    # ni, nx, nt = volume.shape
    shape, _ = utils.get_shape(volume, cigvis.is_line_first())
    nt = shape[2]
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
        clim = utils.auto_clim(volume)

    nodes = []
    for axis, p in pos.items():
        for i in p:
            nodes.append(
                VolumeSlice(
                    volume,
                    axis,
                    i,
                    cmap,
                    clim,
                    nancolor=nancolor,
                    **kwargs,
                ))

    return nodes


def add_mask(nodes: List,
             volumes: Union[List, np.ndarray],
             clims: Union[List, Tuple] = None,
             cmaps: Union[str, List] = None,
             alpha=None,
             excpt=None,
             **kwargs) -> List:
    """
    Add Mask/Overlay volumes
    
    Parameters
    -----------
    nodes: List[Node]
        A List that contains `AxisAlignedImage` (may be created by `create_slices`)
    volumes : array-like or List
        3D array(s), foreground volume(s)/mask(s)
    clims : List
        [vmin, vmax] for foreground slices plotting
    cmaps : str or Colormap
        colormap for foreground slices, it can be str or matplotlib's Colormap or vispy's Colormap
    alpha : float or List[float]
        if alpha is not None, using `colormap.fast_set_cmap` to set cmap
    excpt : None or str
        it could be one of [None, 'min', 'max', 'ramp']

    Returns
    -------
    slices_nodes : List
        list of slice nodes
    """
    if not isinstance(volumes, List):
        volumes = [volumes]

    for volume in volumes:
        # TODO: check shape as same as base image
        utils.check_mmap(volume)

    if clims is None:
        clims = [utils.auto_clim(v) for v in volumes]
    if not isinstance(clims[0], (List, Tuple)):
        clims = [clims]

    if cmaps is None:
        raise ValueError("'cmaps' cannot be 'None'")
    if not isinstance(cmaps, List):
        cmaps = [cmaps] * len(volumes)
    if not isinstance(alpha, List):
        alpha = [alpha] * len(volumes)
    if not isinstance(excpt, List):
        excpt = [excpt] * len(volumes)
    for i in range(len(cmaps)):
        cmaps[i] = colormap.get_cmap_from_str(cmaps[i])
        if alpha[i] is not None:
            cmaps[i] = colormap.fast_set_cmap(cmaps[i], alpha[i], excpt[i])

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


def create_well_logs(
    logs: Union[List, np.ndarray],
    logs_type: str = 'point',
    cmap: str = 'jet',
    clim: List = None,
    width: float = 1,
    point_shape: str = 'square',
    **kwargs,
):
    """
    create well logs nodes

    Parameters
    ----------
    logs : List or array-like
        List (multi-logs) or np.ndarray (one log). For a log,
        its shape is like (N, 3) or (N, 4) or (N, 6) or (N, 7),
        the first 3 columns are (x, y, z) coordinates. If 3 columns,
        use the third column (z) as the color value (mapped by `cmap`), 
        if 4 columns, the 4-th column is the color value (mapped by `cmap`),
        if 6 or 7 columns, colors are RGB format.
    logs_type : str
        'point' or 'line', draw points or line segments
    cmap : str
        colormap for logs
    clim : List
        [vmin, vmax] of logs
    width : float
        width of line segments or points
    point_shape : str
        point shape for points, 'square', 'circle' or others, only when logs_type is 'point'
    
    """
    if not isinstance(logs, List):
        logs = [logs]

    nodes = []
    for log in logs:
        assert log.ndim == 2 and log.shape[1] in [3, 4, 6, 7]
        points = log[:, :3]
        values = None
        colors = None
        if log.shape[1] == 3:
            values = log[:, 2]
        elif log.shape[1] == 4:
            values = log[:, 3]
        else:
            colors = log[:, 3:]

        if logs_type == 'line':
            logs = LogLineSegments
        else:
            logs = LogPoints
        nodes.append(
            logs(
                points,
                values,
                colors,
                cmap,
                clim,
                width,
                point_shape=point_shape,
            ))

    return nodes




def _region2image(pts2d, server: viser.ViserServer):
    client = list(server.get_clients().values())[0]

    h, w = client.camera.image_height, client.camera.image_width

    # Convert normalized box to pixel coordinates
    (u0, v0), (u1, v1) = pts2d
    x0, y0 = int(u0 * w), int(v0 * h)
    x1, y1 = int(u1 * w), int(v1 * h)

    # Clamp + order
    lx, rx = sorted((max(0, x0), min(w, x1)))
    ly, ry = sorted((max(0, y0), min(h, y1)))

    # Create white canvas
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw rectangle
    draw.rectangle([(lx, ly), (rx, ry)], outline=(255, 0, 255), width=2)

    # Draw filled circles at each corner (green)
    for px, py in [(lx, ly), (rx, ly), (rx, ry), (lx, ry)]:
        draw.ellipse([(px-4, py-4), (px+4, py+4)], fill=(0, 255, 0))

    return np.array(canvas)


def plot3D(
    nodes,
    axis_scales=[1, 1, 1],
    fov=30,
    look_at=None,
    wxyz=None,
    position=None,
    server=None,
    run_app=True,
    **kwargs,
):
    if server is None:
        server = viser.ViserServer(label='cigvis-viser',
                                   port=8080,
                                   verbose=False)
    server.scene.reset()
    server.gui.reset()

    fov = fov * np.pi / 180

    global Background_image
    Background_image = None

    # update scale of slices
    draw_slices = -1
    init_scale = -1
    for i, node in enumerate(nodes):
        if isinstance(node, VolumeSlice):
            init_scale = node.init_scale
            node.update_scale(axis_scales)
            draw_slices = i

    if init_scale == -1:  # no slices # TODO: for other types, Well logs?
        init_scale = 100
        for node in nodes:
            if isinstance(node, MeshNode):
                init_scale = min(min(node.scale), init_scale)
        init_scale = [init_scale] * 3

    # update scale of meshes
    meshid, logsid = 0, 0
    for node in nodes:
        if isinstance(node, MeshNode):
            node.scale = [s * x for s, x in zip(init_scale, axis_scales)]
            node.name = f'mesh{meshid}'
            meshid += 1
        elif isinstance(node, LogBase):
            node.scale = [s * x for s, x in zip(init_scale, axis_scales)]
            node.name = f'logs{logsid}-{node.base_name}'
            logsid += 1
        node.server = server

    mask_num = len(nodes[draw_slices].masks)

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
    if draw_slices >= 0:
        [vmin, vmax] = utils.auto_clim(nodes[draw_slices].volume)
        if vmin == vmax:
            vmax = vmin + 1
        step = (vmax - vmin) / 100

    def update_clim(vmin, vmax, type, num):
        if vmin >= vmax:
            return
        for node in nodes:
            if type == 'bg':
                if hasattr(node, 'update_clim'):
                    node.update_clim([vmin, vmax])
            elif type == 'fg':
                if hasattr(node, 'update_mask_clim'):
                    node.update_mask_clim([vmin, vmax], num)

    def update_cmap(cmap):
        for node in nodes:
            if hasattr(node, 'update_cmap'):
                if cmap=='pre-set':
                    cmap = None
                node.update_cmap(cmap)
    
    def update_mask_cmap(cmapname, alpha, excpt, num, first):
        if cmapname == 'pre-set':
            cmap = nodes[first]._fg_cmaps_preset[num]
        else:
            cmap = cmapname
        cmap = colormap.fast_set_cmap(cmap, alpha, excpt)

        for node in nodes:
            if hasattr(node, 'update_mask_cmap'):
                node.update_mask_cmap(cmap, num)

    with server.gui.add_folder("paramters"):
        if draw_slices >= 0:
            guiclim = server.gui.add_vector2('clim', initial_value=tuple(nodes[draw_slices].clim), step=step)
            guiclim.on_update(lambda _: update_clim(*guiclim.value, 'bg', -1))

            guicmap = server.gui.add_dropdown(
                'cmap',
                options=[
                    'pre-set', 'gray', 'seismic', 'Petrel', 'stratum', 'jet', 'bwp'
                ],
                initial_value='pre-set',
            )
            guicmap.on_update(lambda _: update_cmap(guicmap.value))

            if mask_num > 0:
                step1 = (nodes[draw_slices].fg_clims[0][1] - nodes[draw_slices].fg_clims[0][0] + 1e-6) / 100
                maskclim1 = server.gui.add_vector2('mask_clim1', initial_value=tuple(nodes[draw_slices].fg_clims[0]), step=step1)
                maskclim1.on_update(lambda _: update_clim(*maskclim1.value, 'fg', 0))
                maskcmap1 = server.gui.add_dropdown('mask_cmap1', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                alpha1 = nodes[draw_slices].fg_cmaps[0](0.5)[-1]
                maskalpha1 = server.gui.add_slider('mask_alpha1', min=0, max=1, step=0.05, initial_value=alpha1)
                excpt1 = nodes[draw_slices].fg_cmaps[0].excpt if hasattr(nodes[draw_slices].fg_cmaps[0], 'excpt') else 'none'
                maskexcpt1 = server.gui.add_dropdown('mask_excpt1', options=['none', 'min', 'max', 'ramp'], initial_value=excpt1)
                maskcmap1.on_update(lambda _: update_mask_cmap(maskcmap1.value, maskalpha1.value, maskexcpt1.value, 0, draw_slices))
                maskalpha1.on_update(lambda _: update_mask_cmap(maskcmap1.value, maskalpha1.value, maskexcpt1.value, 0, draw_slices))
                maskexcpt1.on_update(lambda _: update_mask_cmap(maskcmap1.value, maskalpha1.value, maskexcpt1.value, 0, draw_slices))

            if mask_num > 1:
                step2 = (nodes[draw_slices].fg_clims[1][1] - nodes[draw_slices].fg_clims[1][0] + 1e-6) / 100
                maskclim2 = server.gui.add_vector2('maskclim2', initial_value=tuple(nodes[draw_slices].fg_clims[1]), step=step2)
                maskclim2.on_update(lambda _: update_clim(*maskclim2.value, 'fg', 1))
                maskcmap2 = server.gui.add_dropdown('mask_cmap2', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                alpha2 = nodes[draw_slices].fg_cmaps[1](0.5)[-1]
                maskalpha2 = server.gui.add_slider('mask_alpha2', min=0, max=1, step=0.05, initial_value=alpha2)
                excpt2 = nodes[draw_slices].fg_cmaps[1].excpt if hasattr(nodes[draw_slices].fg_cmaps[1], 'excpt') else 'none'
                maskexcpt2 = server.gui.add_dropdown('mask_excpt2', options=['none', 'min', 'max', 'ramp'], initial_value=excpt2)
                maskcmap2.on_update(lambda _: update_mask_cmap(maskcmap2.value, maskalpha2.value, maskexcpt2.value, 1, draw_slices))
                maskalpha2.on_update(lambda _: update_mask_cmap(maskcmap2.value, maskalpha2.value, maskexcpt2.value, 1, draw_slices))
                maskexcpt2.on_update(lambda _: update_mask_cmap(maskcmap2.value, maskalpha2.value, maskexcpt2.value, 1, draw_slices))


            if mask_num > 2:
                step3 = (nodes[draw_slices].fg_clims[2][1] - nodes[draw_slices].fg_clims[2][0] + 1e-6) / 100
                maskclim3 = server.gui.add_vector2('maskclim3', initial_value=tuple(nodes[draw_slices].fg_clims[2]), step=step3)
                maskclim3.on_update(lambda _: update_clim(*maskclim3.value, 'fg', 2))
                maskcmap3 = server.gui.add_dropdown('mask_cmap3', options=['pre-set', 'jet', 'stratum', 'Faults', 'gray'], initial_value='pre-set')
                alpha3 = nodes[draw_slices].fg_cmaps[2](0.5)[-1]
                maskalpha3 = server.gui.add_slider('mask_alpha3', min=0, max=1, step=0.05, initial_value=alpha3)
                excpt3 = nodes[draw_slices].fg_cmaps[2].excpt if hasattr(nodes[draw_slices].fg_cmaps[2], 'excpt') else 'none'
                maskexcpt3 = server.gui.add_dropdown('mask_excpt3', options=['none', 'min', 'max', 'ramp'], initial_value=excpt3)
                maskcmap3.on_update(lambda _: update_mask_cmap(maskcmap3.value, maskalpha3.value, maskexcpt3.value, 2, draw_slices))
                maskalpha3.on_update(lambda _: update_mask_cmap(maskcmap3.value, maskalpha3.value, maskexcpt3.value, 2, draw_slices))
                maskexcpt3.on_update(lambda _: update_mask_cmap(maskcmap3.value, maskalpha3.value, maskexcpt3.value, 2, draw_slices))

        # gui to control aspect
        def update_scale(scale):
            for node in nodes:
                if isinstance(node, VolumeSlice):
                    node.update_scale(scale)
                elif isinstance(node, MeshNode):
                    node.scale = [s * x for s, x in zip(init_scale, scale)]

        gui_scale = server.gui.add_vector3('scale', initial_value=(1, 1, 1), step=0.05, min=(0.1, 0.1, 0.1))
        gui_scale.on_update(lambda _: update_scale(gui_scale.value))

    _has_image_height = version.parse(viser.__version__) > version.parse("0.2.23")

    if _has_image_height:
        with server.gui.add_folder("screenshot"):
            select_btn = server.gui.add_button("Select rectangular region")
            show_region = server.gui.add_checkbox("show boundary", False)
            region_left = server.gui.add_vector2("left_up", initial_value=(0, 0), min=(0, 0), max=(1, 1), step=0.001)
            region_right = server.gui.add_vector2("right_down", initial_value=(1, 1), min=(0, 0), max=(1, 1), step=0.001)
            render_btn = server.gui.add_button("Render and get PNG")


    def _select_region(server: viser.ViserServer):
        @server.scene.on_pointer_event(event_type="rect-select")
        def _box(event: viser.ScenePointerEvent) -> None:  # type: ignore[name-defined]
            # event.screen_pos is ((u_min, v_min), (u_max, v_max)), each in [0, 1]
            region_left.value = event.screen_pos[0]
            region_right.value = event.screen_pos[1]
            server.scene.remove_pointer_callback()

    def _draw_render_boundary(server: viser.ViserServer):
        global Background_image
        region = (region_left.value, region_right.value)
        Background_image = _region2image(region, server)
        if not show_region.value:
            return
        server.scene.set_background_image(Background_image, format='png')

    def _update_box(server: viser.ViserServer):
        if not show_region.value:
            server.scene.set_background_image(None)
        else:
            global Background_image
            if Background_image is None:
                region = (region_left.value, region_right.value)
                Background_image = _region2image(region, server)
            server.scene.set_background_image(Background_image, format='png')

    def _render_save(server: viser.ViserServer):
        client = list(server.get_clients().values())[0]

        if show_region.value:
            server.scene.set_background_image(None)

        region = (region_left.value, region_right.value)
        # Full-res render at current camera resolution
        h, w = client.camera.image_height, client.camera.image_width
        image = client.get_render(height=h, width=w, transport_format='png')

        if show_region.value:
            global Background_image
            server.scene.set_background_image(Background_image)

        # Convert normalized box to pixel coordinates
        (u0, v0), (u1, v1) = region
        x0, y0 = int(u0 * w), int(v0 * h)
        x1, y1 = int(u1 * w), int(v1 * h)

        # Clamp + order
        x0, x1 = sorted((max(0, x0), min(w, x1)))
        y0, y1 = sorted((max(0, y0), min(h, y1)))

        if x1 <= x0 or y1 <= y0:
            print(f"[{client.client_id}] Empty crop; aborting.")
            return
        cropped = image[y0:y1, x0:x1]  # HWC
        # Encode & send
        png_bytes = iio.imwrite("<bytes>", cropped, extension=".png")
        client.send_file_download("selection.png", png_bytes)

    if _has_image_height:
        select_btn.on_click(lambda _: _select_region(server))
        region_left.on_update(lambda _: _draw_render_boundary(server))
        region_right.on_update(lambda _: _draw_render_boundary(server))
        show_region.on_update(lambda _: _update_box(server))
        render_btn.on_click(lambda _: _render_save(server))

    with server.gui.add_folder("states"):
        gui_camera = server.gui.add_text("camera", "fov", multiline=True)
        gui_camera_update = server.gui.add_button("get_current_camera")
        gui_states = server.gui.add_button('print states')

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.fov = fov  # Or some other angle in radians, np.pi / 6 -> 30 degree
        if look_at is None:
            client.camera.look_at = (1, 1, 0)
        else:
            client.camera.look_at = tuple(look_at)
        if wxyz is not None:
            client.camera.wxyz = wxyz
        if position is not None:
            client.camera.position = tuple(position)
        gui_camera.value = _fmt_camera_text(client.camera)

    server.scene.set_up_direction((0.0, 0.0, -1.0))

    def _update_camera(server, text):
        client = list(server.get_clients().values())[-1]
        try:
            fov, look_at, wxyz, position = _parser_camera_text(text)
            fov = fov * np.pi / 180
        except Exception:
            return
        client.camera.fov = fov
        client.camera.look_at = look_at
        client.camera.wxyz = wxyz
        client.camera.position = position

    def _get_states(server: viser.ViserServer):
        client = list(server.get_clients().values())[0]

        camera = client.camera
        text = _fmt_camera_text(camera)
        gui_camera.value = text

    def _print_states(server: viser.ViserServer):
        client = list(server.get_clients().values())[0]

        camera = client.camera
        print('')
        print('----------- Current States ------------')
        print(f'fov: {_round(camera.fov * 180 / np.pi)}')
        print(f'look_at: {_round(camera.look_at)}')
        print(f'wxyz: {_round(camera.wxyz)}')
        print(f'position: {_round(camera.position)}')
        print('----------- axis position -------------')
        print(f'x: {guix.value}, y: {guiy.value}, z: {guiz.value}')
        print('----------- parameters ----------------')
        print(f'cmap: {guicmap.value}')
        print(f'clim: {guiclim.value}')
        if mask_num > 0:
            print(f'mask_clim1: {maskclim1.value}')
            print(f'mask_cmap1: {maskcmap1.value}')
            print(f'mask_alpha1: {maskalpha1.value}')
            print(f'mask_excpt1: {maskexcpt1.value}')
        if mask_num > 1:
            print(f'mask_clim2: {maskclim2.value}')
            print(f'mask_cmap2: {maskcmap2.value}')
            print(f'mask_alpha2: {maskalpha2.value}')
            print(f'mask_excpt2: {maskexcpt2.value}')
        if mask_num > 2:
            print(f'mask_clim3: {maskclim3.value}')
            print(f'mask_cmap3: {maskcmap3.value}')
            print(f'mask_alpha3: {maskalpha3.value}')
            print(f'mask_excpt3: {maskexcpt3.value}')
        print('----------- Aspect Ratio --------------')
        print(f'scale: {gui_scale.value}') # yapf: disable
        if _has_image_height:
            print('----------- Screenshot Region --------------')
            print(f'left_up: {region_left.value}')
            print(f'right_down: {region_right.value}')

        print('')

        
    gui_camera.on_update(lambda _: _update_camera(server, gui_camera.value))
    gui_camera_update.on_click(lambda _: _get_states(server))
    gui_states.on_click(lambda _: _print_states(server))

    if run_app and not cigvis.is_running_in_notebook():
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            server.stop()
            del server
            print("Execution interrupted")


def run():
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Execution interrupted")


def create_server(port=8080, label='cigvis-viser', verbose=False):
    return viser.ViserServer(label=label, port=port, verbose=verbose)


def _round(f):
    if np.isscalar(f):
        return round(f, 2)
    if isinstance(f, list):
        return [round(x, 2) for x in f]
    if isinstance(f, np.ndarray):
        return np.round(f, 2)

def _fmt_camera_text(camera):
    fov = _round(camera.fov * 180 / np.pi)
    look_at = camera.look_at
    wxyz = camera.wxyz
    position = camera.position
    return f"fov: {fov}, look_at: {look_at}, wxyz: {wxyz}, position: {position}"

def _parser_camera_text(text):
    pattern = r"fov: (.*?), look_at: (.*?), wxyz: (.*?), position: (.*?)$"
    match = re.fullmatch(pattern, text.strip())
    if not match:
        raise ValueError("字符串格式不正确")
    fov_str, look_at_str, wxyz_str, position_str = match.groups()
    try:
        fov = float(fov_str)
    except ValueError:
        raise ValueError("fov 必须是浮点数")
    
    def _parse_float_list(s: str, expected_length: int = None) -> List[float]:
        s = s.strip("[]")
        parts = [x.strip() for x in s.split(" ") if x.strip()]
        try:
            lst = [float(x) for x in parts]
        except ValueError:
            raise ValueError(f"列表元素必须为浮点数，得到: {s}")
        
        if expected_length is not None and len(lst) != expected_length:
            raise ValueError(f"列表长度必须为 {expected_length}，得到: {len(lst)}")
        return lst
    
    look_at = _parse_float_list(look_at_str, 3)
    wxyz = _parse_float_list(wxyz_str, 4)
    position = _parse_float_list(position_str, 3)
    
    return fov, tuple(look_at), tuple(wxyz), tuple(position)
