# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
F3 demo
===========

F3 demo, `https://dataunderground.org/dataset/f3 <https://dataunderground.org/dataset/f3>`_.

Interpretation data is provided by Xinming Wu, all data is big endian.

CIG's profile photo (`https://github.com/USTC-CIG <https://github.com/USTC-CIG>`_)

.. image:: ../../_static/cigvis/more_demos/070.png
    :alt: image
    :align: center
"""
# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/070.png'

import numpy as np
import cigvis


def load_wellLog(p):
    nlog = 4
    npoints = 2121
    x = [259, 619, 339, 141]
    y = [33, 545, 704, 84]
    z = np.arange(0, 0.2 * npoints, 0.2)
    v = np.fromfile(p, np.float32).reshape(nlog, npoints)
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

    root = '/Volumes/T7/DATA/cigvisdata/F3/'
    seisp = root + 'seis.dat'
    saltp = root + 'salt.dat'
    hz2p = root + 'hz.dat'
    uncp = root + 'unc.dat'
    unc2p = root + 'unc2.dat'
    ni, nx, nt = 591, 951, 362
    shape = (ni, nx, nt)

    # seismic slices
    seis = np.memmap(seisp, np.float32, 'c', shape=shape)
    nodes = cigvis.create_slices(seis,
                                 pos=[ni - 2, 25, nt - 2],
                                 cmap='gray',
                                 clim=[-2.0, 1.5])

    # salt (geologic body)
    salt = np.memmap(saltp, np.float32, 'c', shape=shape)
    nodes += cigvis.create_bodys(salt, 0.0, 0.0, color='cyan')

    # hrizon (surface)
    hz2 = np.fromfile(hz2p, np.float32).reshape(ni, nx)
    nodes += cigvis.create_surfaces([hz2], value_type='yellow')

    # displacement field of unconformity (volume)
    unc = np.fromfile(uncp, np.float32).reshape(shape).astype(np.float32)
    # unconformity (surface)
    unc2 = np.fromfile(unc2p, np.float32).reshape(ni, nx).astype(np.float32)
    nodes += cigvis.create_surfaces([unc2], volume=unc, value_type='amp')

    # well logs
    nodes += load_wellLog(root + 'logs.dat')

    # fault skin
    nodes += cigvis.create_fault_skin(root + 'skins/',
                                      endian='>',
                                      values_type='likelihood')

    cigvis.plot3D(nodes,
                  azimuth=-65.0,
                  elevation=22.0,
                  fov=15.0,
                  axis_scales=(1, 1, 1.7),
                  zoom_factor=1.4)
