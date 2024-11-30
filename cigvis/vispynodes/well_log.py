# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
This file is modified from vispy's TubeVisual

Create a tube like well log
"""

from typing import List
from vispy.visuals.mesh import MeshVisual
from vispy.scene.visuals import Compound
import numpy as np
from numpy.linalg import norm
from vispy.util.transforms import rotate

import collections


class WellLog(Compound):
    """
    Well log Mesh, the trajectory of the well log is shown as 
    a tube (with `radius`). The first curve is shown as the color of
    the tube, and the other curves are shown as faces attach to the tube

    Parameters
    ----------
    points : array-like
        the trajectory of the well log, shape is like (N, 3)
    radius : int or List
        radius of the tube and the curves width
    colors : array-like
        shape is like (N, 4) (one curve, colors for the tube), or 
        (m, N, 4) (m curves)
    index : List
        points index of each curve attached to
    tube_points : int
        the number of points to represent a circle
    mode : str
        default is 'triangles'
    """

    def __init__(self,
                 points,
                 radius,
                 colors,
                 index=None,
                 tube_points=16,
                 mode='triangles',
                 cmap=None,
                 clim=None,
                 shading='smooth',
                 dyn_light=True):
        assert tube_points > 2
        assert points.ndim == 2 and points.shape[1] == 3

        # make sure we are working with floats
        points = np.array(points).astype(float)
        radius = self._process_radius(radius, len(points))
        assert len(radius) < tube_points

        if colors.ndim == 2:
            colors = colors[np.newaxis, ...]
        assert colors.shape == (len(radius), len(points), 4)

        if index is None and len(radius) > 1:
            n = len(radius) - 1
            index = np.arange(0, tube_points, tube_points // n)
        if index is not None:
            index = np.array(index)[:len(radius) - 1]
            assert max(index) < tube_points and min(index) >= 0

        self._cmap = cmap
        self._clim = clim
        self.dyn_light = dyn_light
        tangents, normals, binormals = _frenet_frames(points, False)

        # tube mesh
        tube_vertices, tube_indices = make_triangle_tube(
            points, radius[0], tube_points, normals, binormals)
        tube_colors = colors[0, ...]
        tube_colors = np.repeat(tube_colors, tube_points, axis=0)

        tube = MeshVisual(tube_vertices,
                          tube_indices,
                          vertex_colors=tube_colors,
                          shading=shading,
                          mode=mode)

        line_facemesh = []
        # line face mesh
        for i in range(len(radius) - 1):
            vertices = tube_vertices[index[i]::tube_points]
            assert len(vertices) == len(points)
            line_vertices, line_indieces = make_triangle_line(
                points, radius[i + 1], tube_points, index[i], vertices,
                normals, binormals)
            line_colors = np.vstack([colors[i + 1, ...], colors[i + 1, ...]])
            # line_colors = ColorArray(line_colors)

            # don't add shading
            line_facemesh.append(
                MeshVisual(line_vertices,
                           line_indieces,
                           vertex_colors=line_colors,
                           shading=None,
                           mode=mode))

        self.meshs = [tube] + line_facemesh
        Compound.__init__(self, self.meshs)

    def _process_radius(self, radius, N):
        if not isinstance(radius, List):
            radius = [radius]

        for i in range(len(radius)):
            # if single radius, convert to list of radii
            if not isinstance(radius[i], collections.abc.Iterable):
                radius[i] = [radius[i]] * N
            elif len(radius[i]) != N:
                raise ValueError(
                    f'Length of radii list must match points. Error index: {i}'
                )

        return radius

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap

    @property
    def clim(self):
        return self._clim

    @clim.setter
    def clim(self, clim):
        self._clim = clim


def make_triangle_tube(points: np.ndarray,
                       radius,
                       tube_points: int,
                       normals=None,
                       binormals=None):
    """
    make triangle tube, return vertices and faces
    """
    assert points.shape[1] >= 3

    points = points[:, :3]

    if normals is None or binormals is None:
        tangents, normals, binormals = _frenet_frames(points, False)
    segments = len(points) - 1

    if not isinstance(radius, (float, int, np.number)):
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


def make_triangle_line(points,
                       radius,
                       tube_points,
                       vidx,
                       vertices,
                       normals=None,
                       binormals=None):
    """
    make triangle curve, return vertices and faces
    """

    points = points[:, :3]

    if normals is None or binormals is None:
        tangents, normals, binormals = _frenet_frames(points, False)
    segments = len(points) - 1

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


def _frenet_frames(points, closed=False):
    """
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
        if norm(vec) > epsilon:
            vec /= norm(vec)
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
