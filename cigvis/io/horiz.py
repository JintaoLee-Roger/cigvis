import numpy as np
import re


def convert(hz, ni, nx, dt, istart, xstart, tstart=0):
    hz = np.array([
        list(map(float,
                 s.split('\t')[1:])) for s in hz.split('\n') if s.strip()
    ])
    sorted_indices = np.lexsort((hz[:, 1], hz[:, 0]))
    hz = hz[sorted_indices]
    x = hz[:, 0].astype(int) - istart
    y = hz[:, 1].astype(int) - xstart
    hz[:, 2] = (hz[:, 2] - tstart) / dt

    grid = np.full((ni, nx), -1.0)
    valid_indices = (x < ni) & (y < nx)
    grid[x[valid_indices], y[valid_indices]] = hz[valid_indices, 2]
    grid = grid.astype(np.float32)

    return grid
