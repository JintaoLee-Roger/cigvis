import numpy as np
import cigvis
import matplotlib.pyplot as plt


def blending_color(bg,
                   fg,
                   alpha: float,
                   bgc='gray',
                   fgc='jet',
                   bgclims=None,
                   fgclims=None):
    if bgclims is None:
        bgclims = [bg.min(), bg.max()]
    if fgclims is None:
        fgclims = [fg.min(), fg.max()]
    norm1 = plt.Normalize(vmin=bgclims[0], vmax=bgclims[1])
    norm2 = plt.Normalize(vmin=fgclims[0], vmax=fgclims[1])

    cmap_rgb1 = plt.get_cmap(bgc)
    cmap_rgb2 = plt.get_cmap(fgc)
    arr1_rgb = cmap_rgb1(norm1(bg))
    arr2_rgb = cmap_rgb2(norm2(fg))

    blended_rgb = arr1_rgb * (1 - alpha) + arr2_rgb * alpha

    return blended_rgb



# prepare seismic volume, surface and fg array
volume = .... # ni, nx, nt
surf = .... # ni, nx
fg = ... # ni, nx

# interpolate surface value, shape is (ni, nx)
bg = cigvis.utils.surfaceutils.interp_surf(volume, surf)

# shape is (ni, nx, 3)
color = blending_color(bg, fg, alpha=0.5, bgc='gray', 
                       fgc='jet', bgclims=..., fgclims=...)

# shape is (ni, nx, 4)
tomesh = np.concatenate([surf[:, :, np.newaxis], bg], axis=2)

nodes = cigvis.create_surfaces(tomesh, value_type='value')