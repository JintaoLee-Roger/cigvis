# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Functions for drawing 3D seismic figure using plotly
----------------------------------------------------

TODO: The code for the `plotly` part is not yet fully developed,
and there are only some basic implementations.
We will continue to improve it in the future.

Note
----
Only run in jupyter environment (not include Ipython)

In plotly, for a seismic volume,
- x means inline order
- y means crossline order
- z means time order

- ni means the dimension size of inline / x
- nx means the dimension size of crossline / y
- nt means the dimension size of time / depth / z


Examples
---------
>>> volume.shape = (192, 200, 240) # (ni, nx, nt)

\# only slices
>>> slice_traces = create_slices(volume, pos=[0, 0, 239], cmap='Petrel', show_cbar=True)
>>> plot3D(slice_traces)

\# add surface

\# surfs = [surf1, surf2, ...], each shape is (ni, nx)
>>> sf_traces = create_surfaces(surfs, surf_color='depth')

\# or use amplitude as color
>>> sf_traces = create_surfaces(surfs, volume, surf_color='amp')
>>> plot3D(slice_traces+sf_traces)

For more and detail examples, please refer our documents
"""
import warnings
from typing import List, Tuple, Union, Dict
import copy
import numpy as np
from cigvis import ExceptionWrapper
try:
    import plotly.graph_objects as go
except BaseException as e:
    go = ExceptionWrapper(
        e,
        "run `pip install \"cigvis[plotly]\"` or run `pip install \"cigvis[all]\"` to enable jupyter support"
    )

from skimage.measure import marching_cubes
from skimage import transform

import cigvis
from cigvis import colormap
from cigvis.utils import plotlyutils
import cigvis.utils as utils


def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  scale: float = 1,
                  show_cbar: bool = False,
                  cbar_params: Dict = None):
    """
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
        colormap, it can be str or matplotlib's Colormap or vispy's Colormap
    show_bar : bool
        show colorbar
    cbar_params : Dict
        parameters pass to colorbar

    Returns
    -------
    traces : List
        List of go.Surface
    """
    line_first = cigvis.is_line_first()

    shape = volume.shape
    nt = shape[2] if line_first else shape[0]

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
    vmin, vmax = clim

    slices, pos = plotlyutils.make_slices(volume, pos=pos)

    dimname = dict(x='inline', y='crossline', z='time')

    cmap = colormap.cmap_to_plotly(cmap)

    traces = []

    idx = 0
    for dim in ['x', 'y', 'z']:

        assert len(slices[dim]) == len(pos[dim])

        for j in range(len(slices[dim])):

            if show_cbar and idx == 0:
                showscale = True
            else:
                showscale = False

            idx += 1

            s = slices[dim][j]
            if scale != 1:
                s = transform.resize(
                    s, (s.shape[0] // scale, s.shape[1] // scale),
                    3,
                    anti_aliasing=True)
            # plotlyutils.verifyshape(s.shape, shape, dim)
            num = pos[dim][j]
            name = f'{dim}/{dimname[dim]}'
            xx, yy, zz = plotlyutils.make_xyz(num, shape, dim, s.shape)

            traces.append(
                go.Surface(x=xx,
                           y=yy,
                           z=zz,
                           surfacecolor=s,
                           colorscale=cmap,
                           cmin=vmin,
                           cmax=vmax,
                           name=name,
                           colorbar=cbar_params,
                           showscale=showscale,
                           showlegend=False))

    return traces


def create_overlay(bg_volume: np.ndarray,
                   fg_volume: np.ndarray,
                   pos: Union[List, Dict] = None,
                   bg_clim: List = None,
                   fg_clim: List = None,
                   bg_cmap: str = 'Petrel',
                   fg_cmap: str = None,
                   show_cbar: bool = False,
                   cbar_type: str = 'fg',
                   **kwargs):
    """
    """

    # check
    utils.check_mmap(bg_volume)
    if not isinstance(fg_volume, List):
        fg_volume = [fg_volume]
    for volume in fg_volume:
        assert bg_volume.shape == volume.shape
        utils.check_mmap(volume)

    shape = bg_volume.shape
    line_first = cigvis.is_line_first()
    if line_first:
        nt = bg_volume.shape[2]
    else:
        nt = bg_volume.shape[0]

    # set pos
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

    if bg_clim is None:
        bg_clim = utils.auto_clim(bg_volume)
    if fg_clim is None:
        fg_clim = [utils.auto_clim(v) for v in fg_volume]
    if not isinstance(fg_clim[0], (List, Tuple)):
        fg_clim = [fg_clim]

    assert fg_cmap is not None
    if not isinstance(fg_cmap, (List, Tuple)):
        fg_cmap = [fg_cmap]

    rpos = pos
    bg_slices, pos = plotlyutils.make_slices(bg_volume, pos=rpos)
    fg_slices = []
    for volume in fg_volume:
        s, _ = plotlyutils.make_slices(volume, pos=rpos)
        fg_slices.append(s)

    dimname = dict(x='inline', y='crossline', z='time')

    traces = []

    for dim in ['x', 'y', 'z']:

        assert len(bg_slices[dim]) == len(pos[dim])

        for j in range(len(bg_slices[dim])):

            num = pos[dim][j]
            name = f'{dim}/{dimname[dim]}'
            xx, yy, zz = plotlyutils.make_xyz(num, shape, dim)
            x, y, z, ii, jj, kk = plotlyutils.make_triang(xx, yy, zz)

            # blending
            bs = bg_slices[dim][j]
            plotlyutils.verifyshape(bs.shape, shape, dim)
            fs = [fg[dim][j] for fg in fg_slices]
            colors = colormap.arrs_to_image([bs] + fs, [bg_cmap] + fg_cmap,
                                            [bg_clim] + fg_clim,
                                            True).reshape(-1, 4)
            # colors = np.round(colors * 255).reshape(-1, 3)
            cplotly = [f'rgb({x[0]}, {x[1]}, {x[2]})' for x in colors]

            traces.append(
                go.Mesh3d(x=x,
                          y=y,
                          z=z,
                          i=ii,
                          j=jj,
                          k=kk,
                          name=name,
                          vertexcolor=cplotly))

    if show_cbar:
        fg_cmap = colormap.cmap_to_plotly(fg_cmap[-1])
        fg_clim = fg_clim[-1]
        tickvals = np.linspace(fg_clim[0], fg_clim[1], 6)
        ticktext = [f'{i/1e6:.2f}M' for i in tickvals]
        cbar = go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                colorscale=fg_cmap,  # 与 vertexcolor 对应的 colorscale
                cmin=fg_clim[0],  # 手动设置 colorscale 范围
                cmax=fg_clim[1],
                colorbar=dict(title="Seismic Amplitude",
                              titleside="right",
                              tickvals=tickvals,
                              ticktext=ticktext,
                              ticks="outside",
                              thickness=15,),
                showscale=True))
        traces.append(cbar)

    return traces


def create_surfaces(
    surfs,
    volume=None,
    value_type='depth',
    clim=None,
    cmap='jet',
    show_cbar=False,
    **kwargs,
):

    line_first = cigvis.is_line_first()

    # add surface
    if not isinstance(surfs, List):
        surfs = [surfs]

    surfs_values = []
    if value_type == 'amp':
        if volume is None:
            print("Must input volume if value_type is 'amp' (amplitude)")
        for s in surfs:
            surfs_values.append(
                cigvis.utils.surfaceutils.interp_surf(volume, s))
    else:
        surfs_values = copy.deepcopy(surfs)

    if clim is None:
        vmin = min([s.min() for s in surfs_values])
        vmax = max([s.max() for s in surfs_values])
    else:
        vmin, vmax = clim

    # cmap = colormap.cmap_to_plotly(cmap)

    traces = []

    for s, v in zip(surfs, surfs_values):
        if line_first:
            s = s.T
            v = v.T

        traces.append(
            go.Surface(
                z=s,
                surfacecolor=v,
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                showscale=show_cbar,
                # flatshading=False,
                # 光照效果
                lighting=dict(ambient=0.1,
                              diffuse=0.9,
                              specular=0.5,
                              roughness=0.3,
                              fresnel=0.5),

                # 光源位置
                lightposition=dict(x=100, y=200, z=300)))

    return traces


def create_Line_logs(logs, cmap='jet', line_width=8):
    """
    logs can be a np.ndarray (one log), or List of np.ndarray (muti-logs).
    each element's shape is (N, 3) or (N, 4).
    each row is (x, y, z) or (x, y, z, value)
    """

    cmap = colormap.cmap_to_plotly(cmap)

    if isinstance(logs, np.ndarray):
        logs = [logs]

    traces = []
    for log in logs:
        assert log.shape[1] >= 3
        if log.shape[1] == 3:
            value = log[:, 2]
        else:
            value = log[:, 3]

        traces.append(
            go.Scatter3d(x=log[:, 0],
                         y=log[:, 1],
                         z=log[:, 2],
                         line=dict(color=value,
                                   colorscale=cmap,
                                   width=line_width),
                         mode='lines',
                         showlegend=False))

    return traces


def create_well_logs(*args, **kwargs):
    """
    use Mesh3D to create tube logs
    """
    raise NotImplementedError(
        "`create_well_logs` currently not supported in the jupyter, please run it with a .py file. If you must run in jupyter, please consider use `create_Line_logs`"
    )  # noqa: E501


def add_mask(*args, **kwargs):
    raise NotImplementedError(
        "`add_mask` currently not supported in the jupyter, please run it with a .py file. If you must run in jupyter, please consider use `create_overlay`"
    )  # noqa: E501


def create_points(points, color='red', size=3, sym='square'):
    points = np.array(points)

    trace = go.Scatter3d(x=points[:, 0],
                         y=points[:, 1],
                         z=points[:, 2],
                         mode='markers',
                         marker=dict(symbol=sym,
                                     size=size,
                                     color=color,
                                     line=dict(width=1, color='black')),
                         showlegend=False)

    return [trace]


def create_bodys(volume, level, margin: float = None, color='yellow'):
    if margin is not None:
        if isinstance(volume, np.memmap):
            assert volume.mode != 'r', "margin will modify the volume, set `mode='c'` instead of `mode='r'` in np.memmap"
        volume[0, :, :] = margin
        volume[:, 0, :] = margin
        volume[:, :, 0] = margin
        volume[volume.shape[0] - 1, :, :] = margin
        volume[:, volume.shape[1] - 1, :] = margin
        volume[:, :, volume.shape[2] - 1] = margin

    verties, faces, _, _1 = marching_cubes(volume, level)
    x = verties[:, 0]
    y = verties[:, 1]
    z = verties[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    trace = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color=color,
        showscale=False,
        flatshading=False,
        # 光照效果
        lighting=dict(ambient=0.1,
                      diffuse=0.9,
                      specular=0.5,
                      roughness=0.3,
                      fresnel=0.5),

        # 光源位置
        lightposition=dict(x=100, y=200, z=300))

    return [trace]


def create_fault_skin(*args, **kwargs):
    raise NotImplementedError(
        "`add_mask` currently not supported in the jupyter, please run it with a .py file. If you must run in jupyter, please consider use `create_overlay`"
    )  # noqa: E501


def plot3D(traces, **kwargs):

    size = kwargs.get('size', (900, 900))
    size = (size, size) if isinstance(size, (int, np.integer)) else size

    scene = kwargs.get('scene', {})
    scened = plotlyutils.make_3Dscene(**kwargs)
    for k, v in scened.items():
        scene.setdefault(k, v)

    fig = go.Figure(data=traces)

    fig.update_layout(
        height=size[0],
        width=size[1],
        scene=scene,
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False,
    )

    savequality = kwargs.get('savequality', 1)

    fig.show(
        config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'custom_image',
                'scale': savequality
            }
        })
