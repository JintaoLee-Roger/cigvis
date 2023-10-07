# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
Merge all mesh to one mesh
"""

from typing import List
import numpy as np


def merge_meshs(vertices, faces):
    """
    Merge all meshs to one mesh
    """
    assert isinstance(vertices, List)
    assert isinstance(faces, List)
    assert len(vertices) == len(faces)

    num = [len(v) for v in vertices]
    vertices = np.vstack(vertices)

    out_faces = np.empty((0, 3), faces[0].dtype)
    offset = 0
    for i in range(len(num)):
        out_faces = np.vstack([out_faces, faces[i] + offset])
        offset += num[i]
    
    return vertices, out_faces
