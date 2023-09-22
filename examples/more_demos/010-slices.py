"""
三维数据的显示
======================

"""
import numpy as np
import cigvis
from cigvis import colormap


def show(d):
    # show1
    nodes1 = cigvis.create_slices(d)

    # add colorbar, method1, not recommand
    nodes2, cbar = cigvis.create_slices(d, return_cbar=True)
    nodes2.append(cbar)

    # add colorbar, method2, recommand
    nodes3 = cigvis.create_slices(d, cmap='Petrel', clim=[d.min(), d.max()])
    cbar = cigvis.create_colorbar(cmap='Petrel', clim=[d.min(), d.max()])
    nodes3.append(cbar)

    cigvis.plot3D([nodes1, nodes2, nodes3],
                  grid=(1, 3),
                  size=(1600, 500),
                  cbar_region_ratio=0.15)


if __name__ == '__main__':
    root = '../../'
    sxp = root + 'data/co2/sx.dat'
    ni, nx, nt = 192, 192, 240

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))

    show(sx)