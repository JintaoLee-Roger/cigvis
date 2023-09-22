# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Functions for plotting 1D/2D data using matplotlib
----------------------------------------------------

Some 1D and 2D plotting tools (based on matplotlib). 
Since matplotlib is already very convenient 
and provides more freedom, the implementations in this package 
are very simple and are only for reference. 

In the future, we will implement some demos that have more 
geophysical features.
"""

from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm

import cigvis
from cigvis import colormap

################## For traces plot #############


def plot1d(data: np.ndarray or List,
           dt: float = 1,
           beg: float = 0,
           orient: str = 'v',
           figsize: Tuple or List = (2, 8),
           title: str = None,
           axis_label: str = None,
           value_label: str = None,
           fill_up=None,
           fill_down=None,
           fill_color=None,
           c='#1f77b4',
           save=None,
           show=True,
           dpi=600,
           ax=None):
    """
    plot a 1d trace

    Parameters
    -----------
    data : array-like
        input data
    dt : float
        interval of data, such as 0.2 means data sampling in 0.2, 0.4, ...
    beg : float
        begin sampling, beg=1.6, dt=0.2 means data sampling is 1.6, 1.8, ..
    orient : Optinal ['v', 'h']
        orientation of the data, 'v' means vertical, 'h' means horizontal
    figsize : Tuple or List
        (value-axis length, sampling-axis length)
    title : str
        title
    axis_label : str
        sampling-axis label
    value_label : str
        value axis label
    c : mpl.colors.Color
        color for the line
    """
    assert len(data.shape) == 1
    figsize = figsize

    if fill_up is not None:
        assert fill_up > 0 and fill_up < 1
    if fill_down is not None:
        assert fill_down > 0 and fill_down < 1
    fill_color = c if fill_color is None else fill_color

    sampling = np.arange(beg, beg + len(data) * dt, dt)[:len(data)]

    if orient == 'h':
        data, sampling = sampling, data
        value_label, axis_label = axis_label, value_label
        figsize = (figsize[1], figsize[0])

    show = show and ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data, sampling, c=c)
    _fill_traces(ax, data, sampling, fill_down, fill_up, orient, fill_color)
    ax.set_ylabel(axis_label)
    ax.set_xlabel(value_label)
    ax.set_title(title)
    if orient == 'v':
        plt.gca().invert_yaxis()

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.01, dpi=dpi)

    if show:
        plt.tight_layout()
        plt.show()


def plot_multi_traces(data,
                      dt=0.002,
                      beg=0,
                      c='black',
                      fill_up=None,
                      fill_down=None,
                      fill_color='black',
                      figsize=None,
                      xlabel='Trace number',
                      ylabel='Time / s',
                      save: str = None,
                      show: bool = True,
                      dpi=600,
                      ax=None):
    """
    data.shape = (h, n_traces)
    """
    h, n = data.shape
    r = data.max() - data.min()
    r = r if r != 0 else 1
    y = np.arange(beg, beg + h * dt, dt)[:h]

    if fill_up is not None:
        assert fill_up > 0 and fill_up < 1
    if fill_down is not None:
        assert fill_down > 0 and fill_down < 1

    # figsize = figsize if figsize is not None else (4, 6)
    show = show and ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    prev_max = 0
    tick_pos = []
    for i in range(n):
        l = (data[:, i] - data.min()) / r + prev_max
        prev_max = l.max() if data[:, i].sum() != 0 else prev_max + 1
        tick_pos.append(l.mean())
        ax.plot(l, y, c=c)
        _fill_traces(ax, l, y, fill_down, fill_up, color=fill_color)

    tick_label = np.arange(0, n)
    step = 1 if n // 10 == 0 else n // 10
    ax.set_xticks(tick_pos[::step])
    ax.set_xticklabels(tick_label[::step])
    ax.invert_yaxis()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # plt.xticks(fontproperties='Times New Roman', size=tick_size)
    # plt.yticks(fontproperties='Times New Roman', size=tick_size)
    # fontdict = {
    #         'family': 'Times New Roman',
    #         'weight': 'bold',
    #         'size': label_size
    #     }
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.01, dpi=dpi)

    if show:
        plt.tight_layout()
        plt.show()


def _fill_traces(ax, x, y, fill_down, fill_up, orient='v', color='black'):
    h = len(y)
    if fill_down is not None:
        if orient == 'v':
            fmin = x.mean() - (x.max() - x.min()) / 2 * fill_down
            ax.fill_betweenx(y,
                             x, [fmin] * h,
                             where=(x < fmin),
                             interpolate=True,
                             color=color)
        else:
            fmin = y.mean() - (y.max() - y.min()) / 2 * fill_down
            ax.fill_between(x,
                            y, [fmin] * h,
                            where=(y < fmin),
                            interpolate=True,
                            color=color)
    if fill_up is not None:
        if orient == 'v':
            fmax = x.mean() + (x.max() - x.min()) / 2 * fill_up
            ax.fill_betweenx(y,
                             x, [fmax] * h,
                             where=(x > fmax),
                             interpolate=True,
                             color=color)
        else:
            fmax = y.mean() + (y.max() - y.min()) / 2 * fill_up
            ax.fill_between(x,
                            y, [fmax] * h,
                            where=(y > fmax),
                            interpolate=True,
                            color=color)


############## for image plot #####################


def fg_image_args(img,
                  cmap='jet',
                  clim=None,
                  alpha=None,
                  interpolation='bicubic',
                  show_cbar=True,
                  **kwargs):
    """"""
    if cigvis.is_line_first():
        img = img.T

    kw = {}
    kw['X'] = img
    kw['cmap'] = cmap
    if clim is None:
        clim = [img.min(), img.max()]
    kw['vmin'] = clim[0]
    kw['vmax'] = clim[1]
    kw['alpha'] = alpha
    kw['interpolation'] = interpolation
    kw['show_cbar'] = show_cbar
    kw.update(kwargs)

    return [kw]


def line_args(x,
              y,
              color=None,
              alpha=None,
              lw=None,
              label=None,
              marker=None,
              linestyle=None,
              markersize=None,
              **kwargs):
    """"""
    kw = dict(x=x,
              y=y,
              color=color,
              alpha=alpha,
              lw=lw,
              label=label,
              marker=marker,
              linestyle=linestyle,
              markersize=markersize)
    kw.update(kwargs)
    return [kw]


def marker_args(x,
                y,
                s=None,
                c=None,
                marker=None,
                cmap=None,
                norm=None,
                vmin=None,
                vmax=None,
                alpha=None,
                zorder=2,
                **kwargs):
    """"""
    kw = dict(x=x,
              y=y,
              s=s,
              c=c,
              marker=marker,
              cmap=cmap,
              norm=norm,
              vmin=vmin,
              vmax=vmax,
              alpha=alpha,
              zorder=zorder)
    kw.update(kwargs)
    return [kw]


def annotate_args(x, y, text, w=1, h=1, **kwargs):
    """"""
    assert len(x) == len(y)
    assert len(x) == len(text)
    kw = []
    for i in range(len(x)):
        args = dict(text=text[i], xy=(x[i], y[i]), xytext=(x[i] + w, y[i] + h))
        args.update(kwargs)
        kw.append(args)
    return kw


def _check_is_disceret(data):
    size = int(data.size * (1 / 100.0))
    sampled = np.random.choice(data.ravel(), size=size, replace=False)
    unique = np.unique(sampled)
    if len(unique) > 20:
        return False
    else:
        return True


def discrete_cbar(im, ticks_label=None, remove_trans=False):
    cmap = im.cmap
    alpha = im._alpha
    if alpha is not None:
        cmap = colormap.set_alpha(cmap, alpha, False)

    if not _check_is_disceret(im._A):
        raise RuntimeError(
            "You data is not disceret, set `discrete=False` or " +
            "discerete data `show_cbar=True` instead")

    values = np.unique(im._A)
    if ticks_label is not None:
        assert len(values) == len(ticks_label)

    clim = [im._norm._vmin, im._norm._vmax]
    colors = colormap.get_colors_from_cmap(cmap, clim, values)

    if remove_trans:
        indices = np.where(colors[:, -1] == 0)[0]
        colors = colors[~np.isin(np.arange(len(colors)), indices)]
        values = values[~np.isin(np.arange(len(values)), indices)]
        if ticks_label is not None:
            ticks_label = [
                ticks_label[i] for i in range(len(ticks_label))
                if i not in indices
            ]

    cmap = ListedColormap(colors)
    # norm of equal intervals
    norm = BoundaryNorm(np.arange(0, (len(colors) + 1) * 2, 2), cmap.N)
    # set ticks in the center of boundaries
    ticks = np.arange(1, len(colors) * 2, 2)
    if ticks_label is None:
        ticks_label = values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    return sm, ticks, ticks_label,


def plot2d(
        img: np.ndarray,
        fg: Dict = None,
        cmap: str = 'gray',
        clim: List = None,
        interpolation: str = 'bicubic',

        # fig and label
        figsize: Tuple = None,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,

        # axis
        aspect: str = 'equal',
        axisoff: bool = False,
        xsample: List = None,
        ysample: List = None,

        # cbar
        cbar: str = None,
        discrete: bool = False,
        tick_labels: List = None,
        remove_trans: bool = True,

        # legend
        show_legend: bool = False,

        # font size
        xlabel_size: float = None,
        ylabel_size: float = None,
        title_size: float = None,
        ticklabels_size: float = None,
        cbar_ticklabels_size: float = None,
        cbar_label_size: float = None,

        # save
        save: str = None,
        dpi: float = 600,
        show: bool = True,
        ax=None):
    """
    plot2d

    Parameters
    -----------
    img : array-like
        background img, shape=(n1, n2)
    fg : Dict
        foreground, it can be a combination of 
        'img', 'line', 'marker', 'annotate'
    cmap : str or Colormap
        cmap for background image ('img')
    clim : List
        [vmin, vmax] for background
    interpolation : str 
        interpolation method for background
    
    figsize : Tuple
        figure size
    title : str
        title string
    xlabel : str
        label for x axis
    ylabel : str
        label for y axis
    
    aspect : str or float
        can be 'equal' (default), 'auto' or a float number,
        if is a float number v, it means H / W = w
    axisoff : bool
        turn off axis
    xsample : List
        sampling information for x axis, it can be [x_start, x_step].
        x_start is the start of axis, x_step is interval of x axis, 
    ysample : List
        sampling information for y axis, it can be [y_start, y_step].
        y_start is the start of axis, y_step is interval of y axis, 

    cbar : str
        If is not None, will show a colorbar at the right, cabr is 
        the label. If `cbar=''`, will ignore colorbar label
    discrete : bool
        discrete colorbar
    tick_labels : List
        If discrete, set a label for each value
    remove_trans : bool
        If alpha of the value is 0, colorbar will remove the value
    
    show_legend : bool
        show legend if fg contains 'line'
    
    font_size : float
        each font size
    
    save : str
        save name
    show : bool
        call plt.show()
    ax : Any
        parent axes, used when plt.subplots()
    """
    if cigvis.is_line_first():
        img = img.T

    if clim is None:
        clim = [img.min(), img.max()]

    show = show and ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(cmap, str):
        cmap = colormap.get_cmap_from_str(cmap)

    cbar_im = ax.imshow(img,
                        cmap=cmap,
                        vmin=clim[0],
                        vmax=clim[1],
                        aspect=aspect,
                        interpolation=interpolation)

    if fg is not None:
        for k, v in fg.items():
            if k == 'img':
                for i in range(len(v)):
                    show_cbar = v[i].pop('show_cbar', False)
                    im = ax.imshow(aspect=aspect, **v[i])
                    if show_cbar:
                        cbar_im = im
            elif k == 'line':
                for i in range(len(v)):
                    x = v[i]['x']
                    y = v[i]['y']
                    v[i].pop('x', None)
                    v[i].pop('y', None)
                    ax.plot(x, y, **v[i])
            elif k == 'marker':
                for i in range(len(v)):
                    ax.scatter(**v[i])
            elif k == 'annotate':
                for i in range(len(v)):
                    ax.annotate(**v[i])

    if cbar is not None:
        divider = make_axes_locatable(ax)
        box_apspect = None
        if isinstance(aspect, float) and aspect < 1:
            box_apspect = 20 * aspect
        cax = divider.append_axes('right',
                                  size='5%',
                                  pad=0.05,
                                  box_aspect=box_apspect)

        if not discrete:
            cb = plt.colorbar(cbar_im, cax=cax)
        else:
            sm, ticks, tick_labels = discrete_cbar(cbar_im, tick_labels,
                                                   remove_trans)
            cb = plt.colorbar(sm, cax=cax)
            cb.set_ticks(ticks)
            cb.set_ticklabels(tick_labels)

        cb.ax.set_ylabel(cbar, fontsize=cbar_label_size)
        cb.ax.tick_params(labelsize=cbar_ticklabels_size)

    if show_legend:
        plt.legend()

    if xsample is not None:
        xtick = ax.get_xticks()
        xtick = xtick[xtick >= 0]
        xtick = xtick[xtick < img.shape[1]]
        xticklabels = xsample[0] + xtick * xsample[1]
        ax.set_xticks(xtick)
        ax.set_xticklabels(xticklabels)

    if ysample is not None:
        ytick = ax.get_yticks()
        ytick = ytick[ytick >= 0]
        ytick = ytick[ytick < img.shape[0]]
        yticklabels = ysample[0] + ytick * ysample[1]
        ax.set_yticks(ytick)
        ax.set_yticklabels(yticklabels)

    if ticklabels_size is not None:
        ax.tick_params(axis='both', labelsize=ticklabels_size)

    if axisoff:
        ax.axis('off')

    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ylabel_size)

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.01, dpi=dpi)

    if show:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pass