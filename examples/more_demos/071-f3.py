# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
F3 demo2
===========

F3 demo, `https://dataunderground.org/dataset/f3 <https://dataunderground.org/dataset/f3>`_.

Interpretation data is provided by Xinming Wu, all data is big endian

.. image:: ../../_static/cigvis/more_demos/071.png
    :alt: image
    :align: center
"""
# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/071.png'

import numpy as np
import cigvis
from cigvis import colormap


def load_wellLog(p):
    nlog = 4
    npoints = 2121
    x = [259, 619, 339, 141]
    y = [33, 545, 704, 84]
    z = np.arange(0, 0.2 * npoints, 0.2)
    v = np.fromfile(p, '>f4').reshape(nlog, npoints)
    v = 0.5 * np.log(v)

    nodes = []
    for i in range(nlog):
        points = np.c_[np.ones(npoints) * x[i], np.ones(npoints) * y[i], z]
        nodes += cigvis.create_well_logs(points,
                                         v[i],
                                         cyclinder=False,
                                         radius_tube=[2, 5],
                                         null_value=-999.25)

    return nodes


if __name__ == '__main__':

    root = '/Users/lijintao/Downloads/F3/f3d/data/'
    seisp = root + 'seis.dat'
    saltp = root + 'salt.dat'
    hz1p = root + 'hz1.dat'
    hz2p = root + 'hz2.dat'
    unc1p = root + 'unc1.dat'
    unc2p = root + 'unc2.dat'
    ni, nx, nt = 591, 951, 362
    shape = (ni, nx, nt)

    # seismic
    seis = np.memmap(seisp, '>f4', 'c', shape=shape)
    # overlay
    inter = np.memmap(root + 'intp.dat', '>f4', 'c', shape=shape)

    fg_cmap = colormap.set_alpha('jet', 0.6)
    fg_clim = [inter.max() * 0.15, inter.max() * 0.5]
    nodes = cigvis.create_overlay(seis,
                                  inter,
                                  pos=[ni - 2, 25, nt - 2],
                                  bg_cmap='gray',
                                  bg_clim=[-2.0, 1.5],
                                  fg_cmap=fg_cmap,
                                  fg_clim=fg_clim,
                                  fg_interpolation='nearest')

    salt = np.memmap(saltp, '>f4', 'c', shape=shape)
    nodes += cigvis.create_bodys(salt, 0.0, 0.0, color='cyan')

    # hz1 = np.fromfile(hz1p, '>f4').reshape(ni, nx)
    hz2 = np.fromfile(hz2p, '>f4').reshape(ni, nx)
    nodes += cigvis.create_surfaces([hz2], color='yellow')

    unc = np.fromfile(root + 'unc.dat',
                      '>f4').reshape(shape).astype(np.float32)
    # unc1 = np.fromfile(unc1p, '>f4').reshape(ni, nx).astype(np.float32)
    unc2 = np.fromfile(unc2p, '>f4').reshape(ni, nx).astype(np.float32)
    nodes += cigvis.create_surfaces([unc2], volume=unc, value_type='amp')

    nodes += load_wellLog(root + 'logs.dat')

    nodes += cigvis.create_fault_skin(root + 'skins/')

    cigvis.plot3D(nodes,
                  azimuth=-65.0,
                  elevation=22.0,
                  fov=15.0,
                  axis_scales=(1, 1, 1.7),
                  zoom_factor=1.4)
