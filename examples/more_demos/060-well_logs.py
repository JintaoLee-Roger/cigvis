"""
logging track
================

.. image:: ../../_static/cigvis/more_demos/060.png
    :alt: image
    :align: center

"""

# sphinx_gallery_thumbnail_path = '_static/cigvis/more_demos/060.png'

import numpy as np
import cigvis
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent


def show(sx, points):
    """
    显示将测井显示为一条等半径的tube,
    并显示为一个颜色
    """

    nodes1 = cigvis.create_slices(sx)
    nodes1 += cigvis.create_well_logs(points, cmap='orange', radius_tube=3)
    """
    显示将测井显示为一条等半径的tube,
    颜色为深度
    """

    nodes2 = cigvis.create_slices(sx)
    nodes2 += cigvis.create_well_logs(points, cmap='jet', radius_tube=3)

    cigvis.plot3D([nodes1, nodes2],
                  grid=(1, 2),
                  zoom_factor=4,
                  savename='example.png')


if __name__ == '__main__':
    sxp = root / 'data/co2/sx.dat'
    lxp = root / 'data/co2/lx.dat'
    lasp = root / 'data/cb23.las'
    ni, nx, nt = 192, 192, 240

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))
    lx = np.memmap(lxp, np.float32, 'c', shape=(ni, nx, nt))

    # create a well log
    las = cigvis.io.load_las(lasp)
    idx = las['Well']['name'].index('NULL')
    null_value = float(las['Well']['value'][3])
    lasdata = las['data'][:, 1:]

    x = np.linspace(50, 100, len(lasdata))
    y = np.linspace(50, 150, len(lasdata))
    z = np.sin((y - 50) / 200 * np.pi) * 200
    points = np.c_[x, y, z]

    show(sx, points)
