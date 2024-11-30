"""
Surfaces (N, 3) displayed
==========================

.. Note::

    You may feel a lag when rotating, this is due to a bug in vispy. You have two ways to fix it.
    - turn off the changing light: ```cigvis.plot3D(..., dyn_light=False)```.
    - See this pull: https://github.com/vispy/vispy/pull/2532
 

.. image:: ../../_static/cigvis/more_demos/041.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/041.png'

import numpy as np
import cigvis
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent


def show(sx, sfs):
    """
    sf 除了可以是二维的矩阵, 也可以是 (N, 3) 的点矩阵(无序), 
    每个 sf 都是独立的, 即 sfs 里面可以同时包含
    (ni, nx) 的规则网格 和 (N, 3) 的点.
    N 可以不等于 ni * nx
    """
    sfs1 = [sfs[0].copy(), sfs[1].copy()]
    y, x = np.meshgrid(np.arange(192), np.arange(192))
    sfs1[1] = np.c_[x.flatten(), y.flatten(), sfs1[1].flatten()]

    nodes1 = cigvis.create_slices(sx)
    nodes1 += cigvis.create_surfaces(sfs1,
                                     volume=sx,
                                     value_type='amp',
                                     cmap='Petrel',
                                     clim=[sx.min(), sx.max()])
    """
    当 sf.shape = (N, 3) 时, 可以选择是否启用插值(默认不启用)
    下面是启用插值的结果
    """
    sfs2 = [sfs[0].copy(), sfs[1].copy()]
    y, x = np.meshgrid(np.arange(192), np.arange(192))
    sfs2[1] = np.c_[x.flatten(), y.flatten(), sfs2[1].flatten()]
    k = np.random.choice(np.arange(2000, len(sfs2[1])), 30000, False)
    sfs2[1] = sfs2[1][k, :]

    nodes2 = cigvis.create_slices(sx)
    nodes2 += cigvis.create_surfaces(sfs2,
                                     volume=sx,
                                     value_type='amp',
                                     cmap='Petrel',
                                     clim=[sx.min(), sx.max()],
                                     interp=True)
    """
    当 sf.shape = (N, 3) 时, 可以选择是否启用插值(默认启用)
    下面是不启用插值的结果
    """
    sfs3 = [sfs[0].copy(), sfs[1].copy()]
    y, x = np.meshgrid(np.arange(192), np.arange(192))
    sfs3[1] = np.c_[x.flatten(), y.flatten(), sfs3[1].flatten()]
    k = np.random.choice(np.arange(2000, len(sfs3[1])), 30000, False)
    sfs3[1] = sfs3[1][k, :]

    nodes3 = cigvis.create_slices(sx)
    nodes3 += cigvis.create_surfaces(sfs3,
                                     volume=sx,
                                     value_type='amp',
                                     cmap='Petrel',
                                     clim=[sx.min(), sx.max()],
                                     interp=False)
    """
    添加控制点
    """
    nodes4 = cigvis.create_slices(sx)
    nodes4 += cigvis.create_surfaces(sfs,
                                     volume=sx,
                                     value_type='amp',
                                     cmap='Petrel',
                                     clim=[sx.min(), sx.max()])

    points = [[70, 50, 158], [20, 100, 80]]
    nodes4 += cigvis.create_points(points, r=3)

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

    show(sx, [sf1, sf2])  # [x, y, z]
