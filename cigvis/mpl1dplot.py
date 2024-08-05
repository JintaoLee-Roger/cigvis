# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Functions for plotting 1D data using matplotlib
----------------------------------------------------

Some 1D plotting tools (based on matplotlib). 
Since matplotlib is already very convenient 
and provides more freedom, the implementations in this package 
are very simple and are only for reference. 

In the future, we will implement some demos that have more 
geophysical features.
"""

from typing import List, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from cigvis import colormap

################## For traces plot #############


def plot1d(data: Union[np.ndarray, List],
           dt: float = 1,
           beg: float = 0,
           orient: str = 'v',
           figsize: Union[Tuple, List] = (2, 8),
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
                      inter=1.0,
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
                      lw=1,
                      ax=None):
    """
    data.shape = (h, n_traces)
    """
    assert inter <= 1.0 and inter > 0.0
    inter = 1 - inter
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
        prev_max = l.max() - inter if data[:, i].sum() != 0 else prev_max + 1
        tick_pos.append(l.mean())
        ax.plot(l, y, c=c, lw=lw)
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


def plot_signal_compare(raw,
                        offset_df=None,
                        offset_index=None,
                        with_offset=False,
                        ntstart=0,
                        ntend=6144,
                        show=True,
                        save=None,
                        dpi=600,
                        ax=None):

    show = show and ax is None
    fontdict1 = {'weight': 'normal', 'size': 14}

    i = 0
    temp = raw.copy()
    scale = 10
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)

    x = np.arange(ntstart / 100, ntend / 100, 0.01)
    xtick = list(np.arange(0, 64 + 0.1, 5))
    xtick_label = ['%.0f' % i for i in xtick]

    if with_offset == False:
        for j in range(len(raw)):
            line1, = ax.plot(x,
                             temp[j, 0, :] + scale * j,
                             'black',
                             linewidth=0.5,
                             alpha=0.3)
            line1, = ax.plot(x,
                             temp[j, 1, :] + scale * j,
                             'black',
                             linewidth=0.5,
                             alpha=0.3)
            line1, = ax.plot(x,
                             temp[j, 2, :] + scale * j,
                             'black',
                             linewidth=0.5,
                             alpha=0.3)

        ax.set_ylim(len(raw) * scale + 5, -15)
        ax.set_ylabel('Station', fontdict=fontdict1)

    elif with_offset == True:
        for j in range(len(raw)):
            line1, = ax.plot(x,
                             temp[int(offset_index[j])][0] + offset_df[j],
                             'black',
                             linewidth=0.5,
                             alpha=0.3)
            line1, = ax.plot(x,
                             temp[int(offset_index[j])][1] + offset_df[j],
                             'black',
                             linewidth=0.5,
                             alpha=0.3)
            line1, = ax.plot(x,
                             temp[int(offset_index[j])][2] + offset_df[j],
                             'black',
                             linewidth=0.5,
                             alpha=0.3)

        ax.set_ylim(offset_df.max() + 5, offset_df.min() - 15)
        ax.set_ylabel('Offset (km)', fontdict=fontdict1)

    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlabel('Time (s)', fontdict=fontdict1)
    ax.set_xticks(xtick, xtick_label)
    ax.set_xlim(0, 61.4)

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.01, dpi=dpi)

    if show:
        plt.tight_layout()
        plt.show()


def plot1_with_fill(y,
                    x=None,
                    y2='min',
                    v='y',
                    cmap='jet',
                    orient='h',
                    xlabel='',
                    ylabel='',
                    title='',
                    xlim=None,
                    ylim=None,
                    ax=None):
    show = True
    cmap = colormap.get_cmap_from_str(cmap)

    if ax is not None:
        show = False

    if x is None:
        x = np.arange(y.size)
    assert y.size == x.size

    if isinstance(y2, str):
        if y2 == 'min':
            y2 = float(y.min())
        elif y2 == 'max':
            y2 = float(y.max())
        else:
            raise ValueError("Invalid input of y2")

    if isinstance(y2, (int, float, np.number)):
        y2 = np.array([y2] * x.size)
    if isinstance(y2, List):
        y2 = np.array(y2)

    assert isinstance(y2, np.ndarray) and y2.size == x.size

    if isinstance(v, str):
        if v == 'y':
            v = y
        elif v == 'x':
            v = x
        else:
            raise ValueError("Invalid input of v")
    if isinstance(v, List):
        v = np.array(v)
    assert isinstance(v, np.ndarray) and v.size == x.size
    v = v.reshape(1, -1)

    if ax is None:
        if orient == 'h':
            fig, ax = plt.subplots(figsize=(8, 2))
        else:
            fig, ax = plt.subplots(figsize=(2, 8))

    if orient == 'h':
        ax.plot(x, y, c='black')
        ax.plot(x, y2, c='black')
        ax.plot([x[0], x[0]], [y[0], y2[0]], c='black')
        ax.plot([x[-1], x[-1]], [y[-1], y2[-1]], c='black')
        polygon = ax.fill_between(x, y, y2, lw=0, color='none')
    else:
        ax.plot(y, x, c='black')
        ax.plot(y2, x, c='black')
        ax.plot([y[0], y2[0]], [x[0], x[0]], c='black')
        ax.plot([y[-1], y2[-1]], [x[-1], x[-1]], c='black')
        polygon = ax.fill_betweenx(x, y, y2, lw=0, color='none')
        v = v.reshape(-1, 1)[::-1, :]

    verts = np.vstack([p.vertices for p in polygon.get_paths()])
    extent = [
        verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(),
        verts[:, 1].max()
    ]
    filling = ax.imshow(v, cmap=cmap, aspect='auto', extent=extent)
    filling.set_clip_path(polygon.get_paths()[0], transform=ax.transData)

    _xlim, _ylim = None, None
    if xlim:
        if orient == 'v':
            _ylim = xlim
        else:
            _xlim = xlim

    if ylim:
        if orient == 'v':
            _xlim = ylim
        else:
            _ylim = ylim

    if not _xlim:
        _xlim = ax.get_xlim()
    if not _ylim:
        _ylim = ax.get_ylim()

    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if orient == 'v':
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

    if show:
        plt.tight_layout()
        plt.show()


####### don't call ########


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



if __name__ == "__main__":
    import cigvis
    las = cigvis.io.load_las('data/cb23.las')
    data = las['data']
    h = data[:, 0]
    sp = data[:, 1]
    gr = data[:, 2]
    h = h[gr != -999.25]
    gr = gr[gr != -999.25]
    
    plot1_with_fill(gr, h, orient='v', cmap='SunRise')

