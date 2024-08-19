# Copyright (c) 2024 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

import numpy as np


def north_pointer_mesh(direction, scale=2, style='default', **kwargs):
    if style == 'default':
        return default_north_pointer(direction, scale, **kwargs)
    elif style == 'slb' or style == 'Petrel':  # Schlumberger
        return petrel_north_pointer(direction, scale, **kwargs)

    else:
        raise ValueError(f"Unkown north pointer style: {style}")


def petrel_north_pointer(direction,
                         scale=2,
                         c1=[0.361, 0.788, 0.231],
                         c2=[0.733, 0.153, 0.102],
                         **kwargs):
    v1 = np.array([
        [0, 0, 0],  # Lower left corner of the rectangle
        [1, 0, 0],  # Lower right corner of the rectangle
        [0, 2, 0],  # Upper left corner of the rectangle
        [1, 2, 0],  # Upper right corner of the rectangle
        [-0.5, 2, 0],  # Lower left corner of the arrow (triangle)
        [1.5, 2, 0],  # Lower right corner of the arrow (triangle)
        [0.5, 3, 0],  # Vertex of the arrow (triangle)
    ])
    v2 = v1.copy()
    v2[:, 2] += 0.25
    v3 = v2.copy()
    v3[:, 2] += 0.25
    vertices = np.concatenate([v1, v2, v3], axis=0)
    text_pos = np.array([0.5, 4, 0.5])
    # fmt: off
    faces = np.array([
        [0, 1, 2], [1, 3, 2], [4, 5, 6],
        [0, 7, 8], [8, 1, 0],
        [0, 7, 9], [9, 2, 0],
        [1, 8, 10], [10, 3, 1],
        [4, 11, 9], [9, 2, 4],
        [3, 10, 12], [12, 5, 3],
        [4, 11, 13], [13, 6, 4],
        [6, 13, 12], [12, 5, 6],

        [7, 14, 15], [15, 8, 7],
        [7, 14, 16], [16, 9, 7],
        [8, 15, 17], [17, 10, 8],
        [11, 18, 16], [16, 9, 11],
        [10, 17, 19], [19, 12, 10],
        [11, 18, 20], [20, 13, 11],
        [13, 20, 19], [19, 12, 13],
        [14, 15, 16],  [15, 17, 16], [18, 19, 20],
    ])
    # fmt: on
    face_colors = np.array([c1] * 17 + [c2] * 17)

    vertices = vertices / 8 * scale
    text_pos = text_pos / 8 * scale

    # rotate the north pointer
    angle = np.arctan2(direction[0], direction[1])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    vertices[:, :2] = vertices[:, :2].dot(rotation_matrix)
    text_pos[:2] = text_pos[:2].dot(rotation_matrix)

    return vertices, faces, face_colors, text_pos


def default_north_pointer(direction, scale=2, **kwargs):
    vertices = np.array([
        [0, 3, 0], # 0
        [2, 0, 0], # 1
        [0, -3, 0], # 2
        [-2, 0, 0], # 3
        [0.5, 0.5, 0], # 4
        [0.5, -0.5, 0], # 5
        [-0.5, -0.5, 0], # 6
        [-0.5, 0.5, 0], # 7
        [0, 0, 0.5], # 8
        [0, 0, -0.5], # 9
    ])
    faces = np.array([
        [0, 7, 8],
        [0, 7, 9],

        [1, 4, 8],
        [2, 5, 8],
        [3, 6, 8],

        [1, 4, 9],
        [2, 5, 9],
        [3, 6, 9],

        [0, 8, 4],
        [1, 8, 5],
        [2, 8, 6],
        [3, 8, 7],

        [0, 9, 4],
        [1, 9, 5],
        [2, 9, 6],
        [3, 9, 7],
    ])

    face_colors = np.zeros_like(faces).astype(np.float32)
    face_colors[:2, :] = [1, 0, 0]
    face_colors[2:8, :] = [0.8, 0.8, 0.8]
    face_colors[8:, :] = [0, 0, 0]

    text_pos = np.array([0, 4, 0]).astype(float)

    vertices = vertices / 8 * scale
    text_pos = text_pos / 8 * scale

    # rotate the north pointer
    angle = np.arctan2(direction[0], direction[1])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    vertices[:, :2] = vertices[:, :2].dot(rotation_matrix)
    text_pos[:2] = text_pos[:2].dot(rotation_matrix)

    return vertices, faces, face_colors, text_pos
