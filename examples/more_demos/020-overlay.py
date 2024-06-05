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


def show(bg, fg):
    """
    叠加地震和类别(地震相), alpha=0.5
    """
    fg[fg == 1] = 100
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha(cmap, 0.5)  # Note:

    # nodes1 = cigvis.create_overlay(bg,
    #                                fg,
    #                                fg_cmap=cmap,
    #                                fg_interpolation='nearest')
    nodes1 = cigvis.create_slices(bg)
    nodes1 = cigvis.add_mask(nodes1, fg, cmaps=cmap, interpolation='nearest')

    cbar1 = cigvis.create_colorbar(cmap=cmap,
                                   clim=[fg.min(), fg.max()],
                                   discrete=True,
                                   disc_ticks=[values.astype(int)],
                                   label_str='Facies')
    nodes1.append(cbar1)
    """
    mask 值最小的一类, 但是在colorbar中显示最小的一类(白色)
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_min(cmap, 0.5)  # Note:

    # nodes2 = cigvis.create_overlay(bg,
    #                                fg,
    #                                fg_cmap=cmap,
    #                                fg_interpolation='nearest')
    nodes2 = cigvis.create_slices(bg)
    nodes2 = cigvis.add_mask(nodes2, fg, cmaps=cmap, interpolation='nearest')

    cbar2 = cigvis.create_colorbar(cmap=cmap,
                                   clim=[fg.min(), fg.max()],
                                   discrete=True,
                                   disc_ticks=[values.astype(int)],
                                   label_str='Facies')
    nodes2.append(cbar2)
    """
    mask 最小的一类, 同时 colorbar 也去除 
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_min(cmap, 0.5)  # Note:

    # nodes3 = cigvis.create_overlay(bg,
    #                                fg,
    #                                fg_cmap=cmap,
    #                                fg_interpolation='nearest')
    nodes3 = cigvis.create_slices(bg)
    nodes3 = cigvis.add_mask(nodes3, fg, cmaps=cmap, interpolation='nearest')

    values = values[1:]
    cbar3 = cigvis.create_colorbar(cmap=cmap,
                                   clim=[fg.min(), fg.max()],
                                   discrete=True,
                                   disc_ticks=[values.astype(int)],
                                   label_str='Facies')
    nodes3.append(cbar3)
    """
    mask 最大的一类
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_max(cmap, 0.5)  # Note:

    # nodes4 = cigvis.create_overlay(bg,
    #                                fg,
    #                                fg_cmap=cmap,
    #                                fg_interpolation='nearest')
    nodes4 = cigvis.create_slices(bg)
    nodes4 = cigvis.add_mask(nodes4, fg, cmaps=cmap, interpolation='nearest')

    values = values[:-1]
    cbar4 = cigvis.create_colorbar(cmap=cmap,
                                   clim=[fg.min(), fg.max()],
                                   discrete=True,
                                   disc_ticks=[values],
                                   label_str='Facies')
    nodes4.append(cbar4)
    """
    mask 特定的值, 不是最小或最大, 可以是多个值
    """
    values = np.unique(fg)
    colors = ['red', 'green', 'yellow', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    cmap = colormap.set_alpha_except_values(cmap,
                                            0.5,
                                            clim=[fg.min(), fg.max()],
                                            values=[0, 100])  # Note:

    # nodes5 = cigvis.create_overlay(bg,
    #                                fg,
    #                                fg_cmap=cmap,
    #                                fg_interpolation='nearest')
    nodes5 = cigvis.create_slices(bg)
    nodes5 = cigvis.add_mask(nodes5, fg, cmaps=cmap, interpolation='nearest')

    values = values[values != 0]
    values = values[values != 100]
    cbar5 = cigvis.create_colorbar(cmap=cmap,
                                   clim=[fg.min(), fg.max()],
                                   discrete=True,
                                   disc_ticks=[values],
                                   label_str='Facies')
    nodes5.append(cbar5)

    cigvis.plot3D([nodes1, nodes2, nodes3, nodes4, nodes5],
                  grid=(2, 3),
                  size=(1300, 800),
                  cbar_region_ratio=0.24,
                  savename='example.png')


if __name__ == '__main__':
    root = '../../data/'
    sxp = root + 'seis_h360x600x400.dat'
    lxp = root + 'label_h360x600x400.dat'
    ni, nx, nt = 400, 600, 360

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))
    lx = np.memmap(lxp, np.float32, 'c', shape=(ni, nx, nt))

    show(sx, lx)