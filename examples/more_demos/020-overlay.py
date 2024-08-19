"""
Superimposed display of 3D data (discrete)
=============================================

.. image:: ../../_static/cigvis/more_demos/020.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/020.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent


def show(bg, fg):
    """
    叠加地震和类别(地震相), alpha=0.5
    """
    fg[fg == 1] = 100
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha(cmap, 0.5)  # Note:

    nodes1 = cigvis.create_slices(bg)
    nodes1 = cigvis.add_mask(nodes1, fg, cmaps=cmap, interpolation='nearest')
    nodes1 += cigvis.create_colorbar_from_nodes(
        nodes1,
        label_str='Facies',
        select='mask',
        discrete=True,
        disc_ticks=[values.astype(int)],
    )
    """
    mask 值最小的一类, 但是在colorbar中显示最小的一类(白色)
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_min(cmap, 0.5)  # Note:

    nodes2 = cigvis.create_slices(bg)
    nodes2 = cigvis.add_mask(nodes2, fg, cmaps=cmap, interpolation='nearest')
    nodes2 += cigvis.create_colorbar_from_nodes(
        nodes2,
        label_str='Facies',
        select='mask',
        discrete=True,
        disc_ticks=[values.astype(int)],
    )
    """
    mask 最小的一类, 同时 colorbar 也去除 
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_min(cmap, 0.5)  # Note:

    nodes3 = cigvis.create_slices(bg)
    nodes3 = cigvis.add_mask(nodes3, fg, cmaps=cmap, interpolation='nearest')

    values = values[1:]

    nodes3 += cigvis.create_colorbar_from_nodes(
        nodes3,
        label_str='Facies',
        select='mask',
        discrete=True,
        disc_ticks=[values.astype(int)],
    )
    """
    mask 最大的一类
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_max(cmap, 0.5)  # Note:

    nodes4 = cigvis.create_slices(bg)
    nodes4 = cigvis.add_mask(nodes4, fg, cmaps=cmap, interpolation='nearest')

    values = values[:-1]
    nodes4 += cigvis.create_colorbar_from_nodes(
        nodes4,
        label_str='Facies',
        select='mask',
        discrete=True,
        disc_ticks=[values.astype(int)],
    )
    """
    mask 特定的值, 不是最小或最大, 可以是多个值
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_values(cmap,
                                            0.5,
                                            clim=[fg.min(), fg.max()],
                                            values=[0, 100])

    nodes5 = cigvis.create_slices(bg)
    nodes5 = cigvis.add_mask(nodes5, fg, cmaps=cmap, interpolation='nearest')

    values = values[values != 0]
    values = values[values != 100]
    nodes5 += cigvis.create_colorbar_from_nodes(
        nodes5,
        label_str='Facies',
        select='mask',
        discrete=True,
        disc_ticks=[values],
    )

    cigvis.plot3D([nodes1, nodes2, nodes3, nodes4, nodes5],
                  grid=(2, 3),
                  size=(1300, 800),
                  cbar_region_ratio=0.24,
                  savename='example.png')


if __name__ == '__main__':
    sxp = root / 'data/seis_h360x600x400.dat'
    lxp = root / 'data/label_h360x600x400.dat'
    ni, nx, nt = 400, 600, 360

    sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
    lx = np.fromfile(lxp, np.float32).reshape(ni, nx, nt)

    show(sx, lx)
