# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Tools for colormap
------------------

Including:

- custom colormap: petrel

- cmap to vispy and plotly

"""

from typing import List, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import Colormap as mplColormap
import matplotlib.colors as mcolors
import numpy as np

from vispy.color import Colormap as vispyColormap
from vispy.color import get_colormap as get_vispycmap

from .customcmap import *


def blend_two_arrays(bg, fg, bg_cmap, fg_cmap, bg_clim, fg_clim):
    """
    blend two arrays using their cmap
    """
    if isinstance(bg_cmap, str):
        bg_cmap = get_cmap_from_str(bg_cmap)
    if isinstance(fg_cmap, str):
        fg_cmap = get_cmap_from_str(fg_cmap)
    norm1 = plt.Normalize(vmin=bg_clim[0], vmax=bg_clim[1])
    norm2 = plt.Normalize(vmin=fg_clim[0], vmax=fg_clim[1])
    arr1_rgba = bg_cmap(norm1(bg))
    arr2_rgba = fg_cmap(norm2(fg))

    alpha = arr2_rgba[:, :, 3][..., None]
    out = arr1_rgba[:, :, :3] * (1 - alpha) + arr2_rgba[:, :, :3] * alpha

    return out


def blend_multiple(bg, fg, bg_cmap, fg_cmap, bg_clim, fg_clim):
    """
    """
    out = blend_two_arrays(bg, fg[0], bg_cmap, fg_cmap[0], bg_clim, fg_clim[0])

    for i in range(len(fg) - 1):
        norm = plt.Normalize(vmin=fg_clim[i + 1][0], vmax=fg_clim[i + 1][1])
        cmap = fg_cmap[i + 1]
        if isinstance(cmap, str):
            cmap = get_cmap_from_str(cmap)
        arr_rgba = cmap(norm(fg[i + 1]))
        alpha = arr_rgba[:, :, 3][..., None]
        out = out * (1 - alpha) + arr_rgba[:, :, :3] * alpha

    return out


def get_cmap_from_str(cmap: str, includevispy: bool = False):
    """
    return a Colormap from a cmap string
    
    Parameters
    ----------
    cmap : str
        colormap name string
    includevispy : bool
        include vispy's cmaps
    """
    if cmap in list_custom_cmap():
        cmap = get_custom_cmap(cmap)
    else:
        try:
            cmap = plt.get_cmap(cmap)
        except:
            pass

    if isinstance(cmap, str) and includevispy:
        try:
            cmap = get_vispycmap(cmap)
        except:
            pass

    return cmap


def cmap_to_plotly(cmap):
    """
    convert matplotlib's cmap into plotly's style

    Parameters
    ----------
    cmap : cmap in matplotlib
        matplotlib's cmap
    
    Returns
    -------
    plotly_cmap : List
        colormap used in plotly
    """
    if isinstance(cmap, str):
        import plotly.express as px
        if cmap in px.colors.named_colorscales():
            return cmap
        cmap = get_cmap_from_str(cmap)

    d = cmap(np.linspace(0, 1, 256))
    plotly_cmap = []

    for k in range(256):
        color = f'rgb({int(d[k, 0]*255)},{int(d[k, 1]*255)},{int(d[k, 2]*255)})'
        plotly_cmap.append([k / 255, color])

    return plotly_cmap


def cmap_to_vispy(cmap) -> vispyColormap:
    """
    convert matplotlib's cmap into vispy's style

    Parameters
    ----------
    cmap : cmap in matplotlib
        matplotlib's cmap
    
    Returns
    -------
    vispy_cmap : vispy.color.Colormap
        colormap used in vispy
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap, True)
    if isinstance(cmap, mplColormap):
        colors = cmap(np.arange(cmap.N))
        return vispyColormap(colors)
    if isinstance(cmap, vispyColormap):
        return cmap

    if isinstance(cmap, str):
        raise ValueError('Unknow colormap')


def custom_disc_cmap(values: List, colors: List):
    """
    Custom a discrete colormap from values and colors.
    Like this:

    >>> values: [v1, v2, v3, v4, v5]
    >>> colors: [c1, c2, c3, c4, c5]
    >>> bound: [v1, (v1+v2)/2, (v2+v3)/2, (v3+v4)/2, (v4+v5)/2, v5]
            = [b1, b2, b3, b4, b5, b6]
    >>> [b1, b2]: c1, [b2, b3]: c2, [b3, b4]: c3, [b4, b5]: c4, [b5, b6]: c5 

    Parameters
    ----------
    values : List
        values list, N elements
    colors : List
        the correspanding colors of values, N elements, 
        each elements can be str, tuple, such as 'red', '#7F0000', (0, 0.5, 0)

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        colormap

    Examples
    ----------
    >>> values = [0, 2, 7, 8]
    >>> colors = ['green', '#7f6580', 'blue', (0, 0.8, 0.2)]
    >>> cmap = custom_disc_cmap(values, colors)
    """
    assert len(values) == len(colors)
    colors = [mcolors.to_rgb(c) for c in colors]
    values, colors = [list(c) for c in zip(*sorted(zip(values, colors)))]
    colors.append(colors[-1])

    bound = [values[0]]
    for i in range(len(values) - 1):
        bound.append((values[i] + values[i + 1]) / 2)
    bound.append(values[-1])
    l = bound[-1] - bound[0]

    red = [(0, colors[0][0], colors[0][0])]
    green = [(0, colors[0][1], colors[0][1])]
    blue = [(0, colors[0][2], colors[0][2])]
    for i in range(len(bound) - 1):
        red.append(
            ((bound[i + 1] - bound[0]) / l, colors[i][0], colors[i + 1][0]))
        green.append(
            ((bound[i + 1] - bound[0]) / l, colors[i][1], colors[i + 1][1]))
        blue.append(
            ((bound[i + 1] - bound[0]) / l, colors[i][2], colors[i + 1][2]))

    cdict = {'red': red, 'green': green, 'blue': blue}

    return LinearSegmentedColormap('custom_cmap', cdict)


def get_colors_from_cmap(cmap, clim: List, values: List):
    """
    get colors from a cmap when special vmin, vmax and values

    Parameters
    ----------
    cmap : str or matplotlib.color.Colormap
        input cmap
    clim : List
        [vmin, vmax] to Normalize
    values : List
        list of values

    Returns
    -------
    c : List
        colors at norm(values) locations
    """
    values = np.array(values)
    if isinstance(cmap, str):
        if cmap in list_custom_cmap():
            cmap = get_custom_cmap(cmap)
        else:
            if mcolors.is_color_like(cmap):
                c = np.array([mcolors.to_rgba(cmap)] * len(values))
                c[np.isnan(values), -1] = 0
                return c
            else:
                cmap = plt.get_cmap(cmap)
    if isinstance(cmap, vispyColormap):
        rgba = cmap.colors.rgba
        cmap = LinearSegmentedColormap.from_list('vispy_cmap', rgba)

    norm = mpl.colors.Normalize(clim[0], clim[1])
    locs = norm(values)
    c = cmap(locs)

    return c


def discrete_cmap(cmap, clim, values):
    colors = get_colors_from_cmap(cmap, clim, values)
    cmap = custom_disc_cmap(values, colors)

    return cmap


def ramp(cmap, blow=0, up=1, alpha_min=0, alpha_max=1, forvispy=True):
    """
    Creates a modified colormap from an existing colormap, with adjustable transparency (alpha) levels.
    
    Parameters:
    - cmap: The original colormap to be modified.
    - blow (float, optional): The lower bound of the colormap normalization range. Defaults to 0.
    - up (float, optional): The upper bound of the colormap normalization range. Defaults to 1.
    - alpha_min (float, optional): The minimum alpha (transparency) value to apply. Defaults to 0.
    - alpha_max (float, optional): The maximum alpha (transparency) value to apply. Defaults to 1.
    
    Returns:
    - A new colormap with alpha adjusted from alpha_min to alpha_max within the specified range [blow, up].
    """

    cmap = get_cmap_from_str(cmap)
    slope = (alpha_max - alpha_min) / (up - blow)
    N = cmap.N
    arr = cmap(np.arange(N))
    istart = int(blow * N)
    iend = int(up * N)
    arr[:istart, 3] = alpha_min
    arr[iend:, 3] = alpha_max
    arr[istart:iend, 3] = np.arange(iend - istart) / N * slope + alpha_min
    cmap = ListedColormap(arr)
    if forvispy:
        cmap = cmap_to_vispy(cmap)
    return cmap


def set_alpha(cmap, alpha: float, forvispy: bool = True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap. This function is mainly used in vispy which
    doesn't contain a parameter like `alpha` in matplotlib to set opacity

    Parameters
    ----------
    cmap : str or vispyColormap or mplColormap
        the input cmap
    alpha : float
        opacity
    forvispy : bool
        return a vispy Colormap or matplotlib Colormap

    Returns
    -------
    cmap : str or vispyColormap or mplColormap
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)

    if isinstance(cmap, vispyColormap):
        colors = cmap.colors.rgba
        colors[:, -1] = alpha
        return vispyColormap(colors)
    elif isinstance(cmap, mplColormap):
        cmap = ListedColormap(cmap(np.arange(cmap.N), alpha=alpha))
        if forvispy:
            cmap = cmap_to_vispy(cmap)
        return cmap
    else:
        raise ValueError("unkown cmap")


def set_alpha_except_min(cmap, alpha: float, forvispy: bool = True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap and set the alpha of the min value as 0.
    This means mask the min value when used for a discrete show.
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)

    if isinstance(cmap, vispyColormap):
        colors = cmap.colors.rgba
        colors[:, -1] = alpha
        colors[0, 3] = 0
        return vispyColormap(colors)
    elif isinstance(cmap, mplColormap):
        colors = cmap(np.arange(cmap.N), alpha=alpha)
        colors[0, 3] = 0
        cmap = ListedColormap(colors)
        if forvispy:
            cmap = cmap_to_vispy(cmap)
        return cmap
    else:
        raise ValueError("unkown cmap")


def set_alpha_except_max(cmap, alpha: float, forvispy: bool = True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap and set the alpha of the **max** value as 0.
    This means mask the **max** value when used for a discrete show.
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)

    if isinstance(cmap, vispyColormap):
        colors = cmap.colors.rgba
        colors[:, -1] = alpha
        colors[-1, 3] = 0
        return vispyColormap(colors)
    elif isinstance(cmap, mplColormap):
        colors = cmap(np.arange(cmap.N), alpha=alpha)
        colors[-1, 3] = 0
        cmap = ListedColormap(colors)
        if forvispy:
            cmap = cmap_to_vispy(cmap)
        return cmap
    else:
        raise ValueError("unkown cmap")


def set_alpha_except_values(cmap,
                            alpha: float,
                            clim: List,
                            values: List,
                            forvispy: bool = True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap. And set the alpha of the select values as 0 when clim is applied.
    This means mask the values when used for a discrete show.

    Parameters
    ----------
    cmap : str or vispyColormap or mplColormap
        the input cmap
    alpha : float
        opacity
    clim : List
        [vmin, vmax] for mpl.colors.Normalize
    values : List
        the select values to except (or mask) 
    forvispy : bool
        return a vispy Colormap or matplotlib Colormap

    Returns
    -------
    cmap : str or vispyColormap or mplColormap
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)
    if isinstance(cmap, vispyColormap):
        cmap = ListedColormap(cmap.colors.rgba)
    colors = cmap(np.arange(cmap.N), alpha=alpha)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    index = np.interp(norm(values), np.linspace(0, 1, cmap.N),
                      np.arange(cmap.N))
    ceil = np.ceil(index).astype(int)
    floor = np.floor(index).astype(int)
    colors[ceil, 3] = 0
    colors[floor, 3] = 0

    if forvispy:
        return vispyColormap(colors, interpolation='zero')

    return ListedColormap(colors)


def set_alpha_except_top(cmap, alpha, clim, segm, forvispy=True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap. And set alphas in range `[segm, clim[1]]` to 0
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)
    if isinstance(cmap, vispyColormap):
        cmap = ListedColormap(cmap.colors.rgba)
    colors = cmap(np.arange(cmap.N), alpha=alpha)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    index = np.interp(norm(segm), np.linspace(0, 1, cmap.N),
                      np.arange(cmap.N)).astype(int)
    colors[index:, 3] = 0

    if forvispy:
        return vispyColormap(colors, interpolation='zero')

    return ListedColormap(colors)


def set_alpha_except_bottom(cmap, alpha, clim, segm, forvispy=True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap. And set alphas in range `[clim[0], segm]` to 0
    """
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)
    if isinstance(cmap, vispyColormap):
        cmap = ListedColormap(cmap.colors.rgba)
    colors = cmap(np.arange(cmap.N), alpha=alpha)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    index = np.interp(norm(segm), np.linspace(0, 1, cmap.N),
                      np.arange(cmap.N)).astype(int)
    colors[:index, 3] = 0

    if forvispy:
        return vispyColormap(colors, interpolation='zero')

    return ListedColormap(colors)


def set_alpha_except_ranges(cmap, alpha, clim, r, forvispy=True):
    """
    Set the alpha blending value, between 0 (transparent) and 1 (opaque)
    for a cmap. And set the alpha of the range as 0 when clim is applied.
    This means mask the range of values when used for a discrete show.

    Parameters
    ----------
    r : List
        ranges, like [0, 2] or [[1, 2], [5, 8], ...]
    """
    if not isinstance(r[0], List):
        r = [r]
    assert all([len(c) == 2 for c in r])

    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)
    if isinstance(cmap, vispyColormap):
        cmap = ListedColormap(cmap.colors.rgba)
    colors = cmap(np.arange(cmap.N), alpha=alpha)

    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    index = np.interp(norm(r), np.linspace(0, 1, cmap.N),
                      np.arange(cmap.N)).astype(int)

    for c in index:
        colors[c[0]:c[1], 3] = 0

    if forvispy:
        return vispyColormap(colors, interpolation='zero')

    return ListedColormap(colors)


def list_custom_cmap() -> List:
    """
    avaliable cmap list
    """

    return list(custom_cdict.keys())


def get_custom_cmap(cmap: str) -> LinearSegmentedColormap:
    """
    Get a cmap from a name used in Opendtect software. 
    The supported cmap can be see by `get_opendtect_cmap_list()`
    """
    cmap_list = list_custom_cmap()
    if cmap not in cmap_list:
        raise RuntimeWarning(
            f"cmap {cmap} not in opendtect cmap list {cmap_list}")

    return LinearSegmentedColormap(f'{cmap}', custom_cdict[cmap])


def plot_cmap(cmap, norm: List = None, save: str = None):
    """
    plot a cmap with a norm

    Parameters
    -----------
    cmap : str or Colormap
        colormap
    norm : List
        [vmin, vmax], set value range
    save : str
        name to save
    """
    if norm is None:
        norm = mpl.colors.Normalize(-10, 10)
    if isinstance(norm, List):
        norm = mpl.colors.Normalize(norm[0], norm[1])
    if isinstance(cmap, str):
        cmap = get_cmap_from_str(cmap)

    name = cmap if isinstance(cmap, str) else cmap.name
    fig, ax = plt.subplots(figsize=(10, 1))
    fig.subplots_adjust(bottom=0.5)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax,
                 orientation='horizontal',
                 label=f'{name} colormap')
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.02, dpi=600)
    plt.show()


def plot_all_custom_cmap(norm: List = None, save: str = None, dpi=300):
    """
    plot all custom cmaps with a norm
    """
    if norm is None:
        norm = mpl.colors.Normalize(-10, 10)
    if isinstance(norm, List):
        norm = mpl.colors.Normalize(norm[0], norm[1])
    cmap_list = list_custom_cmap()
    N = len(cmap_list)
    nrows = np.ceil(N / 4).astype(int)

    fig, axs = plt.subplots(nrows, 4, figsize=(15, nrows * 0.9))

    for i in range(nrows):
        for j in range(4):
            idx = j + i * 4
            if idx < N:
                cmap = get_custom_cmap(cmap_list[idx])
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                             cax=axs[i, j],
                             orientation='horizontal',
                             label=f'{cmap_list[idx]} colormap')
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.02, dpi=dpi)

    plt.show()
