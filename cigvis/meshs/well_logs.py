# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
To merge all curves into one Mesh,
To merge all trajectory into one Mesh.
"""

import numpy as np


def trajectory_mesh(points: np.ndarray,
                    radius,
                    tube_points: int,
                    normals=None,
                    binormals=None):
    """
    make the trajectory as a tube, return vertices and faces

    Parameters
    -----------
    points : array-like
        trajectory points, shape is like (N, 3)
    
    """
    assert points.shape[1] >= 3

    points = points[:, :3]

    if normals is None or binormals is None:
        tangents, normals, binormals = _frenet_frames(points, False)
    segments = len(points) - 1

    if not isinstance(radius, (int, float, np.number)):
        radius = np.array(radius).reshape(1, -1, 1)
    points = points[np.newaxis, ...]
    normals = normals[np.newaxis, ...]
    binormals = binormals[np.newaxis, ...]
    vertices = np.empty((tube_points, len(points), 3))
    angles = np.arange(tube_points) / tube_points * 2 * np.pi
    angles = angles.reshape(-1, 1, 1)
    vertices = points + -1. * radius * np.cos(angles) * normals + \
        radius * np.sin(angles) * binormals
    vertices = vertices.transpose(1, 0, 2)
    vertices = vertices.reshape(-1, 3)

    # construct the mesh
    indices = np.empty((segments * 2 * tube_points, 3), dtype=int)
    i, j = np.meshgrid(np.arange(segments),
                       np.arange(tube_points),
                       indexing='ij')
    i = i.flatten()
    j = j.flatten()
    jp = (j + 1) % tube_points
    indices[0::2, :] = np.c_[i * tube_points + j, (i + 1) * tube_points + j,
                             i * tube_points + jp]
    indices[1::2, :] = np.c_[(i + 1) * tube_points + j,
                             (i + 1) * tube_points + jp, i * tube_points + jp]

    # closed
    exdices = []
    for start in [0, len(vertices) - tube_points]:
        for i in range(tube_points - 2):
            j = start + i
            exdices.append([start, j + 1, j + 2])
    indices = np.vstack([indices, np.array(exdices)])

    return vertices, indices


def curves_mesh(points,
                radius,
                tube_points,
                vidx,
                traj_vertices,
                normals=None,
                binormals=None):
    """
    make curves as faces, return vertices and faces

    Parameters
    -----------
    points : array-like
        trajectory points, shape is like (N, 3)
    radius : array-like
        shape is like (m, N, 3)
    tube_points : int
        tube points
    vidx : array-like
        len(vidx) == m
    traj_vertices : array-like
        vertices of the trajectory mesh, shape is like (N*tube_points, 3)
    """

    assert points.shape[1] >= 3

    points = points[:, :3]

    if normals is None or binormals is None:
        tangents, normals, binormals = _frenet_frames(points, False)
    segments = len(points) - 1

    # TODO
    # get the positions of each vertex
    radius = radius[:, np.newaxis]
    angles = vidx / tube_points * 2 * np.pi
    grid = points + -1. * radius * np.cos(angles) * normals + \
        radius * np.sin(angles) * binormals

    start = len(vertices)
    vertices = np.vstack([vertices, grid])

    # each point has two face:
    # [[start + i, i, i + 1], [start + i, i + 1, start + i + 1]]
    indices = np.empty((2 * segments, 3), dtype=int)
    indices[:, 1] = np.repeat(np.arange(0, segments), 2)
    indices[:, 0] = indices[:, 1] + start
    indices[:, 2] = indices[:, 1] + 1
    indices[1::2, 1] += 1
    indices[1::2, 2] += start

    return vertices, indices


###### don't call ########


def _frenet_frames(points, closed=False):
    """
    This function is from vispy.visual.TubeVisual

    Calculates and returns the tangents, normals and binormals for
    the tube.
    """
    tangents = np.zeros((len(points), 3))
    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    # Compute tangent vectors for each segment
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    if not closed:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
    mags = np.sqrt(np.sum(tangents * tangents, axis=1))
    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)

    normals[0] = np.cross(tangents[0], vec)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i - 1]

        vec = np.cross(tangents[i - 1], tangents[i])
        if np.linalg.norm(vec) > epsilon:
            vec /= np.linalg.norm(vec)
            theta = np.arccos(np.clip(tangents[i - 1].dot(tangents[i]), -1, 1))
            normals[i] = rotate(-np.degrees(theta),
                                vec)[:3, :3].dot(normals[i])

    if closed:
        theta = np.arccos(np.clip(normals[0].dot(normals[-1]), -1, 1))
        theta /= len(points) - 1

        if tangents[0].dot(np.cross(normals[0], normals[-1])) > 0:
            theta *= -1.

        for i in range(1, len(points)):
            normals[i] = rotate(-np.degrees(theta * i),
                                tangents[i])[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals


def rotate(angle, axis, dtype=None):
    """
    This function is from vispy.utils.transforms

    The 4x4 rotation matrix for rotation about a vector.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.

    Returns
    -------
    M : ndarray
        Transformation matrix describing the rotation.
    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array(
        [[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
         [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
         [cx * z - y * s, cy * z + x * s, cz * z + c, 0.], [0., 0., 0., 1.]],
        dtype).T
    return M
