# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from typing import List
import numpy as np
from vispy.scene.visuals import Compound
from vispy.visuals import LineVisual, TextVisual


# TODO: add grids or meshs? like plotly?
class Axis3D(Compound):
    """
    3D axis with ticks and labels.

    Parameters
    ------------
    shape : tuple
        The bound of the 3D world
    mode : str
        The mode of the axis, 'box' or 'axis'
    tick_nums : int
        The number of ticks on each axis
    ticks_font_size : int
        The font size of the ticks
    labels_font_size : int
        The font size of the labels
    intervals : list
        The sample intervals of the axis
    starts : list
        The first sample of the axis
    samplings : list[np.ndarray]
        The sample points of the axis, default is None
    axis_pos : list
        Which axis to show ticks? For each axis, it can be 0, 1, 2, 3, 
        representing the starting point of the ticks along the axis.
        0: For 'x' axis -> (0, 0, 0), for 'y' axis -> (0, 0, 0), for 'z' axis -> (0, 0, 0)
        1: For 'x' axis -> (0, 0, nz), for 'y' axis -> (0, 0, nz), for 'z' axis -> (0, ny, 0)
        2: For 'x' axis -> (0, ny, 0), for 'y' axis -> (nx, 0, 0), for 'z' axis -> (nx, 0, 0)
        3: For 'x' axis -> (0, ny, nz), for 'y' axis -> (nx, 0, nz), for 'z' axis -> (nx, ny, 0)
    axis_labels : list
        The labels of the axis
    line_width : int
        The width of the axis line
    ticks_length : int
        The length of each tick
    expand : int
        The expand of the axis
    color : str
        The color of the axis
    ticks_color : tuple
        The color of the ticks
    rotation : list
        The rotation of the axis labels
    """

    def __init__(
            self,
            shape,
            mode='box',
            axis_pos=[3, 3, 1],
            tick_nums=7,
            ticks_font_size=18,
            labels_font_size=20,
            intervals=[1, 1, 1],
            starts=[0, 0, 0],
            samplings=None,
            axis_labels=['Inline', 'Xline', 'Time'],
            line_width=3,
            ticks_length=4,
            expand=1,
            color='black',
            ticks_color=(0.3, 0.3, 0.3),
            # HACK: this need be removed in the future
            rotation=(0, 0, -90),
    ):
        assert len(shape) == 3
        if samplings is not None:
            assert isinstance(samplings, List) and len(samplings) == 3
            self.sp_x, self.sp_y, self.sp_z = samplings
            assert len(self.sp_x) == shape[0] and len(
                self.sp_y) == shape[1] and len(self.sp_z) == shape[2]
        else:
            assert len(intervals) == 3 and len(starts) == 3
            self.sp_x = np.arange(shape[0]) * intervals[0] + starts[0]
            self.sp_y = np.arange(shape[1]) * intervals[1] + starts[1]
            self.sp_z = np.arange(shape[2]) * intervals[2] + starts[2]

        self.auto_change = False
        if isinstance(axis_pos, str) and axis_pos == 'auto':
            self.auto_change = True
            axis_pos = [3, 3, 1]

        self.shape = shape
        self.expand = expand
        assert len(axis_labels) == 3
        self.axis_labels = axis_labels
        self.tick_nums = tick_nums
        self.axis_pos = [-1, -1, -1]
        self.mode = mode
        self.ticks_length = ticks_length
        font_size = int(ticks_font_size * max(shape) * 1.5)
        labels_font_size = int(labels_font_size * max(shape) * 1.5)

        self._line1 = LineVisual(method='gl',
                                 connect='segments',
                                 width=line_width,
                                 color=color,
                                 antialias=True)
        self._ticksx = LineVisual(method='gl',
                                  connect='segments',
                                  width=line_width,
                                  color=ticks_color,
                                  antialias=True)
        self._ticksy = LineVisual(method='gl',
                                  connect='segments',
                                  width=line_width,
                                  color=ticks_color,
                                  antialias=True)
        self._ticksz = LineVisual(method='gl',
                                  connect='segments',
                                  width=line_width,
                                  color=ticks_color,
                                  antialias=True)

        self._textx = TextVisual(color='black',
                                 font_size=font_size,
                                 anchor_x='center',
                                 anchor_y='center',
                                 depth_test=True)
        self._texty = TextVisual(color='black',
                                 font_size=font_size,
                                 anchor_x='center',
                                 anchor_y='center',
                                 depth_test=True)
        self._textz = TextVisual(color='black',
                                 font_size=font_size,
                                 anchor_x='center',
                                 anchor_y='center',
                                 depth_test=True)

        self._textaxis = TextVisual(
            pos=[[0, 0, 0]] * 3,
            color='black',
            bold=True,
            rotation=rotation,  # rotate it through transform?
            font_size=labels_font_size,
            depth_test=True)

        self.xticks = self.get_ticks(shape[0], tick_nums)
        self.yticks = self.get_ticks(shape[1], tick_nums)
        self.zticks = self.get_ticks(shape[2], tick_nums)

        self.eight = np.array([
            [-expand, -expand, -expand],
            [shape[0] + expand, -expand, -expand],
            [shape[0] + expand, shape[1] + expand, -expand],
            [-expand, shape[1] + expand, -expand],
            [-expand, -expand, shape[2] + expand],
            [shape[0] + expand, -expand, shape[2] + expand],
            [shape[0] + expand, shape[1] + expand, shape[2] + expand],
            [-expand, shape[1] + expand, shape[2] + expand],
        ])

        self.update_axis_labels(axis_labels)
        self.update_ticks_labels()
        self.update_ticks_pos(axis_pos)
        self.update_axis()

        nodes = [
            self._line1, self._ticksx, self._ticksy, self._ticksz, self._textx,
            self._texty, self._textz, self._textaxis
        ]

        Compound.__init__(self, nodes)

    def _compute_bounds(self, axis, view):
        return -self.expand, self.shape[axis]

    def update_axis_labels(self, axis_labels):
        assert len(axis_labels) == 3
        self._textaxis.text = axis_labels

    def get_line(self, axis, tick_start):
        assert tick_start >= 0 and tick_start < 4
        n1 = int(tick_start / 2)
        n2 = int(tick_start % 2)
        if axis == 'x':
            n1 = -self.expand if n1 == 0 else self.shape[1] + self.expand
            n2 = -self.expand if n2 == 0 else self.shape[2] + self.expand
            line = [[-self.expand, n1, n2],
                    [self.shape[0] + self.expand, n1, n2]]
        elif axis == 'y':
            n1 = -self.expand if n1 == 0 else self.shape[0] + self.expand
            n2 = -self.expand if n2 == 0 else self.shape[2] + self.expand
            line = [[n1, -self.expand, n2],
                    [n1, self.shape[1] + self.expand, n2]]
        elif axis == 'z':
            n1 = -self.expand if n1 == 0 else self.shape[0] + self.expand
            n2 = -self.expand if n2 == 0 else self.shape[1] + self.expand
            line = [[n1, n2, -self.expand],
                    [n1, n2, self.shape[2] + self.expand]]

        return line

    def update_axis(self):
        if self.mode == 'box' and self._line1.pos is not None:
            return
        if not (self.update_x or self.update_y or self.update_z):
            return
        eight = self.eight
        # fmt: off
        if self.mode == 'box':
            line = [
                eight[0], eight[1],
                eight[1], eight[2],
                eight[2], eight[3],
                eight[3], eight[0],
                eight[0], eight[4],
                eight[1], eight[5],
                eight[2], eight[6],
                eight[3], eight[7],
                eight[4], eight[5],
                eight[5], eight[6],
                eight[6], eight[7],
                eight[7], eight[4],
            ]
        elif self.mode == 'axis':
            line = []
            line += self.get_line('x', self.axis_pos[0])
            line += self.get_line('y', self.axis_pos[1])
            line += self.get_line('z', self.axis_pos[2])
        # fmt: on
        self._line1.set_data(line)

    def update_ticks_pos(self, axis_pos=None, axis=None, tick_start=None):
        """
        Updates the tick positions on a specified axis in a 3D scene.

        Parameters
        -----------
        - axis (str): The axis on which to update the ticks. Should be one of 'x', 'y', or 'z'.
        - tick_start (int): An integer representing the starting point of the ticks along the axis.
            The tick_start corresponds to the following start points:
            0: For 'x' axis -> (0, 0, 0), for 'y' axis -> (0, 0, 0), for 'z' axis -> (0, 0, 0)
            1: For 'x' axis -> (0, 0, nz), for 'y' axis -> (0, 0, nz), for 'z' axis -> (0, ny, 0)
            2: For 'x' axis -> (0, ny, 0), for 'y' axis -> (nx, 0, 0), for 'z' axis -> (nx, 0, 0)
            3: For 'x' axis -> (0, ny, nz), for 'y' axis -> (nx, 0, nz), for 'z' axis -> (nx, ny, 0)
        """
        self.update_x, self.update_y, self.update_z = False, False, False
        if axis_pos is None:
            if axis is None or tick_start is None:
                raise ValueError(
                    "`axis_pos` is None, so must provide both `axis` and `tick_start`"
                )
            assert tick_start >= 0 and tick_start < 4
            if axis == 'x' and tick_start != self.axis_pos[0]:
                self.axis_pos[0] = tick_start
                self.update_x = True
            elif axis == 'y' and tick_start != self.axis_pos[1]:
                self.axis_pos[1] = tick_start
                self.update_y = True
            elif axis == 'z' and tick_start != self.axis_pos[2]:
                self.axis_pos[2] = tick_start
                self.update_z = True
        else:
            assert len(axis_pos) == 3
            assert min(axis_pos) >= 0 and max(axis_pos) < 4
            if axis_pos[0] != self.axis_pos[0]:
                self.axis_pos[0] = axis_pos[0]
                self.update_x = True
            if axis_pos[1] != self.axis_pos[1]:
                self.axis_pos[1] = axis_pos[1]
                self.update_y = True
            if axis_pos[2] != self.axis_pos[2]:
                self.axis_pos[2] = axis_pos[2]
                self.update_z = True

        if self.update_x:
            n1 = int(self.axis_pos[0] / 2)
            n2 = int(self.axis_pos[0] % 2)
            n1 = -self.expand if n1 == 0 else self.shape[1] + self.expand
            n2 = -self.expand if n2 == 0 else self.shape[2] + self.expand
            offset = -self.ticks_length if n1 < 0 else self.ticks_length
            ticks_pos = []
            ticks_label_pos = []
            for x in self.xticks:
                ticks_pos += [[x, n1, n2], [x, n1 + offset, n2]]
                ticks_label_pos.append([x, n1 + offset * 4, n2])
            self._ticksx.set_data(np.array(ticks_pos))
            self._textx.pos = np.array(ticks_label_pos)
            pos = self._textaxis.pos
            pos[0] = [self.shape[0] / 2, n1 + offset * 8, n2]
            self._textaxis.pos = pos
        if self.update_y:
            n1 = int(self.axis_pos[1] / 2)
            n2 = int(self.axis_pos[1] % 2)
            n1 = -self.expand if n1 == 0 else self.shape[0] + self.expand
            n2 = -self.expand if n2 == 0 else self.shape[2] + self.expand
            offset = -self.ticks_length if n1 < 0 else self.ticks_length
            ticks_pos = []
            ticks_label_pos = []
            for y in self.yticks:
                ticks_pos += [[n1, y, n2], [n1 + offset, y, n2]]
                ticks_label_pos.append([n1 + offset * 4, y, n2])
            self._ticksy.set_data(np.array(ticks_pos))
            self._texty.pos = np.array(ticks_label_pos)
            pos = self._textaxis.pos
            pos[1] = [n1 + offset * 8, self.shape[1] / 2, n2]
            self._textaxis.pos = pos
        if self.update_z:
            n1 = int(self.axis_pos[2] / 2)
            n2 = int(self.axis_pos[2] % 2)
            n1 = -self.expand if n1 == 0 else self.shape[0] + self.expand
            n2 = -self.expand if n2 == 0 else self.shape[1] + self.expand
            os1 = -self.ticks_length if n1 < 0 else self.ticks_length
            os2 = -self.ticks_length if n2 < 0 else self.ticks_length
            ticks_pos = []
            ticks_label_pos = []
            for z in self.zticks:
                ticks_pos += [[n1, n2, z], [n1 + os1, n2 + os2, z]]
                ticks_label_pos.append([n1 + os1 * 4, n2 + os2 * 4, z])
            self._ticksz.set_data(np.array(ticks_pos))
            self._textz.pos = np.array(ticks_label_pos)
            pos = self._textaxis.pos
            pos[2] = [n1 + os1 * 8, n2 + os2 * 8, self.shape[2] / 2]
            self._textaxis.pos = pos

    def update_ticks_labels(self):
        def _fmt(x):
            return f'{x:.2f}'.rstrip('0').rstrip('.')
        self.xtick_labels = [_fmt(self.sp_x[i]) for i in self.xticks]
        self.ytick_labels = [_fmt(self.sp_y[i]) for i in self.yticks]
        self.ztick_labels = [_fmt(self.sp_z[i]) for i in self.zticks]
        self._textx.text = self.xtick_labels
        self._texty.text = self.ytick_labels
        self._textz.text = self.ztick_labels

    def get_ticks(self, length, num=7):
        """
        get ticks for axis, num is the number of ticks
        """
        # interval = int(length / (num - 1))
        # return np.arange(0, length + interval, interval)[:num - 1]
        vmin = 0
        vmax =length - 1
        data_range = vmax - vmin
        raw_interval = data_range / (num - 1)
        mag = 10 ** np.floor(np.log10(raw_interval))
        normalized_interval = raw_interval / mag

        if normalized_interval <= 1.5:
            tick_interval = 1 * mag
        elif normalized_interval <= 3:
            tick_interval = 2 * mag
        elif normalized_interval <= 7:
            tick_interval = 5 * mag
        else:
            tick_interval = 10 * mag

        tick = np.arange(0, length, tick_interval).astype(int)
        return tick
