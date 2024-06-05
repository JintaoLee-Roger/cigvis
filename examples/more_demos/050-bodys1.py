"""
Geologic body
==============

.. image:: ../../_static/cigvis/more_demos/050.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/050.png'

import numpy as np
import cigvis


def show1(sx, lx, level):
    nodes = cigvis.create_slices(sx)
    nodes += cigvis.create_bodys(lx, level, margin=0)

    cigvis.plot3D(nodes, savename='example.png')


if __name__ == '__main__':
    sxp = '../../data/co2/sx.dat'
    lxp = '../../data/co2/lx.dat'
    ni, nx, nt = 192, 192, 240

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))
    lx = np.memmap(lxp, np.float32, 'c', shape=(ni, nx, nt))

    show1(sx, lx, 0.5)