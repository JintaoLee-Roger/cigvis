# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import numpy as np


def surf_to_mask(sfs, shape, width=1):

    def fill_with(volume, sf, number):
        assert sf.ndim == 2
        n1, n2, n3 = volume.shape
        sf = np.round(sf).astype(int)
        if sf.ndim == 2 and sf.shape[-1] == 3:
            valid = (sf[:, 0] >= 0) & (sf[:, 0] < n1) & \
                (sf[:, 1] >= 0) & (sf[:, 1] < n2) & \
                    (sf[:, 2] >= 0) & (sf[:, 2] < n3)
            sf = sf[valid]
            for i in range(width):
                j = -width // 2 + i
                volume[sf[:, 0], sf[:, 1], sf[:, 2] + j] = number
        else:
            assert sf.shape == (n1, n2)
            x, y = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
            valid = (sf >= 0) & (sf < n3)
            x, y, z = x[valid], y[valid], sf[valid]
            for i in range(width):
                j = -width // 2 + i
                volume[x, y, z + j] = number
        return volume

    if isinstance(sfs, np.ndarray):
        sfs = [sfs]
    out = np.zeros(shape, np.float32)

    for i, sf in enumerate(sfs):
        fill_with(out, sf, i + 1)

    return out

