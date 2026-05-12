"""
Discrete colorbar
======================

.. image:: ../../_static/cigvis/more_demos/011.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/011.png'

import numpy as np
import cigvis
from cigvis import colormap
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent


def show_discrete_cbar(d):
    """
    Add a discrete colorbar.

    For discrete data, set interpolation to 'nearest'.
    """
    nodes1 = cigvis.create_slices(d, cmap='jet', interpolation='nearest')
    cbar1 = cigvis.create_colorbar(
        cmap='jet',
        clim=[d.min(), d.max()],
        discrete=True,
        disc_ticks=[np.unique(d), ['CA', 'CB', 'CC', 'CD', '100']],
        label_str='Facies')
    nodes1.append(cbar1)
    """
    For discrete data, outlines can make neighboring colors look very similar.
    """
    d[d == 1] = 100
    nodes2 = cigvis.create_slices(d, cmap='jet', interpolation='nearest')
    nodes2 += cigvis.create_colorbar_from_nodes(
        nodes2,
        select='slices',
        discrete=True,
        disc_ticks=[np.unique(d).astype(int)],
        label_str='Facies')
    """
    A custom colormap can fix the issue shown in cbar2.
    """
    values = np.unique(d)
    colors = ['gray', 'green', '#7f6589', 'blue', (0, 0.5, 0.5)]
    cmap = colormap.custom_disc_cmap(values, colors)
    nodes3 = cigvis.create_slices(d, cmap=cmap, interpolation='nearest')
    nodes3 += cigvis.create_colorbar_from_nodes(
        nodes3,
        label_str='Facies',
        select='slices',
        discrete=True,
        disc_ticks=[values.astype(int)])
    """
    Use get_colors_from_cmap to sample colors.

    Then build a custom colormap to fix the issue shown in cbar2.
    """
    values = np.unique(d)

    # Sample evenly spaced colors.
    colors = colormap.get_colors_from_cmap('jet',
                                           clim=[0, len(values) - 1],
                                           values=np.arange(0, len(values)))

    cmap = colormap.custom_disc_cmap(values, colors)
    nodes4 = cigvis.create_slices(d, cmap=cmap, interpolation='nearest')
    nodes4 += cigvis.create_colorbar_from_nodes(
        nodes3,
        label_str='Facies',
        select='slices',
        discrete=True,
        disc_ticks=[values.astype(int)])

    cigvis.plot3D([nodes1, nodes2, nodes3, nodes4],
                  view=cigvis.Plot3DView(
                      grid=(2, 2),
                      cbar_region_ratio=0.18,
                      size=(1400, 1000),
                  ),
                  save=cigvis.Plot3DSave(path='example.png'))


if __name__ == '__main__':
    lxp = root / 'data/label_h360x600x400.dat'
    ni, nx, nt = 400, 600, 360

    lx = np.fromfile(lxp, np.float32).reshape(ni, nx, nt)

    show_discrete_cbar(lx)
