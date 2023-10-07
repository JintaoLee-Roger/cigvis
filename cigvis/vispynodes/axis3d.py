# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from vispy.scene.visuals import Compound
from vispy.visuals import LineVisual
import numpy as np

int_to_axis = {0: 'x', 1: 'y', 2: 'z'}


class InterSectionLine(Compound):
    """"""

    def __init__(self, shape, pos, color='white', interval=0.1, width=1):
        if pos.count(None) == 0:
            self.nlines = 3
        elif pos.count(None) == 1:
            self.nlines = 1
        else:
            raise RuntimeError(f"No InterSection Line for pos: {pos}")

        if self.nlines == 1:
            idx = pos.index(None)
            axis = int_to_axis[idx]
            pos = [v for v in pos if v is not None]
            Compound.__init__(
                self,
                [FourLines(axis, pos, shape[idx], interval, color, width)])
        else:
            lines = []
            for i in range(self.nlines):
                tpos = [v for k, v in enumerate(pos) if k != i]
                lines.append(
                    FourLines(int_to_axis[i], tpos, shape[i], interval, color,
                              width))
            Compound.__init__(self, lines)

    def update_pos(self):
        """"""
        pass


class FourLines(LineVisual):

    def __init__(self,
                 axis,
                 pos,
                 length,
                 interval=0.1,
                 color='white',
                 width=1):
        assert len(pos) == 2
        assert axis in ['x', 'y', 'z']
        self.axis = axis
        self.len = length
        self.int = interval
        # self.antialias = True

        LineVisual.__init__(self, color=color, width=width, antialias=True)
        self._update_location(pos)

    def _update_location(self, pos):
        p = np.array([[0, pos[0] - self.int, pos[1] - self.int],
                      [0, pos[0] - self.int, pos[1] + self.int],
                      [0, pos[0] + self.int, pos[1] - self.int],
                      [0, pos[0] + self.int, pos[1] + self.int],
                      [self.len, pos[0] - self.int, pos[1] - self.int],
                      [self.len, pos[0] - self.int, pos[1] + self.int],
                      [self.len, pos[0] + self.int, pos[1] - self.int],
                      [self.len, pos[0] + self.int, pos[1] + self.int]])
        if self.axis == 'y':
            p = p[:, [1, 0, 2]]
        elif self.axis == 'z':
            p = p[:, [1, 2, 0]]

        points = np.array([
            p[0], p[1], p[3], p[2], p[0], p[4], p[5], p[7], p[6], p[4], p[5],
            p[1], p[3], p[7], p[6], p[2]
        ])
        self.set_data(pos=points)


class BoxLine(Compound):

    def __init__(self,
                 shape,
                 expand=3,
                 color='black',
                 width=1,
                 antialias=True):
        if isinstance(expand, (float, int)):
            expand = [expand] * 3
        assert len(expand) == 3
        assert len(shape) == 3
        shape = np.array(shape)
        expand = np.array(expand)
        p8 = shape + expand
        p1 = -1 * expand
        p2 = [p1[0], p1[1], p8[2]]
        p3 = [p1[0], p8[1], p1[2]]
        p4 = [p1[0], p8[1], p8[2]]
        p5 = [p8[0], p1[1], p1[2]]
        p6 = [p8[0], p1[1], p8[2]]
        p7 = [p8[0], p8[1], p1[2]]

        line1 = LineVisual(np.array([p1, p2, p4, p3, p1, p5, p6, p2]),
                           color=color,
                           width=width,
                           antialias=antialias)
        line2 = LineVisual(np.array([p6, p8, p7, p5]),
                           color=color,
                           width=width,
                           antialias=antialias)
        line3 = LineVisual(np.array([p4, p8]),
                           color=color,
                           width=width,
                           antialias=antialias)
        line4 = LineVisual(np.array([p3, p7]),
                           color=color,
                           width=width,
                           antialias=antialias)

        Compound.__init__(self, [line1, line2, line3, line4])


class Axis3D(Compound):

    def __init__(self):
        # TODO
        pass