"""
Comparison grid
===============

Compare two or three processed volumes with shared dimension/index controls.
"""

import numpy as np

from cigvis import sliceviewer as sv


rng = np.random.default_rng(12)
raw = rng.normal(size=(3, 4, 64, 96)).astype(np.float32)
denoised = 0.65 * raw + 0.35 * raw.mean(axis=0, keepdims=True)
attribute = np.gradient(denoised, axis=-1).astype(np.float32)

nodes_raw = sv.create_slice(raw, display_axes=(2, 3), indices={0: 1, 1: 2})
nodes_denoised = sv.create_slice(denoised, display_axes=(2, 3), indices={0: 1, 1: 2})
nodes_attribute = sv.create_slice(attribute, display_axes=(2, 3), indices={0: 1, 1: 2})


if __name__ == "__main__":
    sv.show(
        [nodes_raw, nodes_denoised, nodes_attribute],
        grid=(2, 2),
        port=5007,
        title="Processing comparison",
    )
