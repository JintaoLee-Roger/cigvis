"""
Surfaces (n1, n2) are displayed
================================

.. Note::

    You may feel a lag when rotating, this is due to a bug in vispy. You have two ways to fix it.
    - turn off the changing light: ```cigvis.plot3D(..., dyn_light=False)```.
    - See this pull: https://github.com/vispy/vispy/pull/2532
 

.. image:: ../../_static/cigvis/more_demos/040.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/040.png'

import numpy as np
import cigvis
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent


def show(sx, sfs):
    """
    颜色为 depth, sf 里面的所有层位使用同一个 clim
    """
    nodes1 = cigvis.create_slices(sx)
    nodes1 += cigvis.create_surfaces(sfs,
                                     volume=sx,
                                     value_type='depth',
                                     clim=[0, 239])
    """
    颜色为 depth, sf 里面的每个层位单独设置 clim
    """
    nodes2 = cigvis.create_slices(sx)
    for sf in sfs:
        nodes2 += cigvis.create_surfaces(sf,
                                         volume=sx,
                                         value_type='depth',
                                         clim=[sf.min(), sf.max()])
    """
    color: amplitude
    """
    # nodes = []
    nodes3 = cigvis.create_slices(sx)
    nodes3 += cigvis.create_surfaces(sfs,
                                     volume=sx,
                                     value_type='amp',
                                     cmap='Petrel',
                                     clim=[sx.min(), sx.max()])
    """
    一个mask的层位, 将二维矩阵中小于0的部分mask
    """
    sfs[1][:120, 50:100] = -1

    nodes4 = cigvis.create_slices(sx)
    nodes4 += cigvis.create_surfaces(sfs,
                                     volume=sx,
                                     value_type='amp',
                                     cmap='Petrel',
                                     clim=[sx.min(), sx.max()])

    cigvis.plot3D([nodes1, nodes2, nodes3, nodes4],
                  grid=(2, 2),
                  share=True,
                  savename='example.png')


if __name__ == '__main__':
    sxp = root / 'data/co2/sx.dat'
    lxp = root / 'data/co2/lx.dat'
    sf1p = root / 'data/co2/mh21.dat'
    sf2p = root / 'data/co2/mh22.dat'
    ni, nx, nt = 192, 192, 240

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))
    lx = np.memmap(lxp, np.float32, 'c', shape=(ni, nx, nt))

    sf1 = np.fromfile(sf1p, np.float32).reshape(ni, nx)
    sf2 = np.fromfile(sf2p, np.float32).reshape(ni, nx)

    show(sx, [sf1, sf2])  # depth
