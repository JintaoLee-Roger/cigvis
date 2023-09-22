"""
三维数据的叠加显示 (连续)
==========================

"""
import numpy as np
import cigvis
from cigvis import colormap


def show(bg, fg, fx):
    """
    seismic and RGT
    """
    cmap1 = colormap.set_alpha('jet', 0.5)
    nodes1 = cigvis.create_overlay(bg,
                                   fg,
                                   pos=[[36], [28], [84]],
                                   bg_cmap='gray',
                                   fg_cmap=cmap1)

    cbar1 = cigvis.create_colorbar(cmap='jet',
                                   clim=[fg.min(), fg.max()],
                                   label_str='RGT')
    nodes1.append(cbar1)
    """
    mask RGT 的底部 (mask fg > 120)
    """
    cmap2 = colormap.set_alpha_except_max('jet', 0.5)
    nodes2 = cigvis.create_overlay(bg,
                                   fg,
                                   pos=[[36], [28], [84]],
                                   bg_cmap='gray',
                                   fg_clim=[fg.min(), 120],
                                   fg_cmap=cmap2)

    # set vmax = 150
    cbar2 = cigvis.create_colorbar(cmap='jet',
                                   clim=[fg.min(), 120],
                                   label_str='RGT')
    nodes2.append(cbar2)
    """
    seismic and fault
    """
    # Note: set alpha = 1 is the best
    cmap3 = colormap.set_alpha_except_min('jet', 1)
    nodes3 = cigvis.create_overlay(bg,
                                   fx,
                                   pos=[[36], [28], [84]],
                                   bg_cmap='gray',
                                   fg_cmap=cmap3,
                                   fg_interpolation='nearest')

    values = np.unique(fx).astype(int)
    values = values[1:]
    labels = ['fx'+str(i) for i in values]
    cbar3 = cigvis.create_colorbar(cmap='jet',
                                   clim=[fx.min(), fx.max()],
                                   label_str='fault',
                                   discrete=True,
                                   disc_ticks=[values, labels])
    nodes3.append(cbar3)
    """
    seismic, masked RGT (mask fg > 120) and fault
    """
    cmap4 = [
        colormap.set_alpha_except_max('jet', 0.5),
        colormap.set_alpha_except_min('jet', 1)
    ]
    interp = ['cubic', 'nearest']
    fg_clim = [[fg.min(), 120], [fx.min(), fx.max()]]
    nodes4 = cigvis.create_overlay(bg, [fg, fx],
                                   pos=[[36], [28], [84]],
                                   bg_cmap='gray',
                                   fg_cmap=cmap4,
                                   fg_clim=fg_clim,
                                   fg_interpolation=interp)

    cbar4 = cigvis.create_colorbar(cmap='jet',
                                   clim=[fg.min(), 120],
                                   label_str='RGT')
    nodes4.append(cbar4)

    cigvis.plot3D([nodes1, nodes2, nodes3, nodes4],
                  grid=(2, 2),
                  size=(1400, 1200),
                  cbar_region_ratio=0.18)


if __name__ == '__main__':
    sxp = '../../data/rgt/sx.dat'
    uxp = '../../data/rgt/ux.dat'
    fxp = '../../data/rgt/fx.dat'
    ni, nx, nt = 128, 128, 128

    sx = np.memmap(sxp, np.float32, 'c', shape=(ni, nx, nt))
    fx = np.memmap(fxp, np.float32, 'c', shape=(ni, nx, nt))
    ux = np.memmap(uxp, np.float32, 'c', shape=(ni, nx, nt))

    show(sx, ux, fx)
