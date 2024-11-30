# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, modified by Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC)
#
#
# Copyright (C) 2019 Yunzhi Shi @ The University of Texas at Austin.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for more info.
# -----------------------------------------------------------------------------

import numpy as np
from vispy.visuals.visual import BaseVisual
from vispy import scene
from vispy.scene.visuals import Mesh, Compound, Line, Text
from vispy.visuals.transforms import MatrixTransform
import cigvis
from cigvis.meshs import north_pointer_mesh


class FixMixin:

    def on_resize(self, event):
        # When window is resized, move the node accordingly.
        loc = np.array(self.loc).astype(np.single)
        loc *= (np.array(event.size).astype(np.single) /
                np.array(self.canvas_size).astype(np.single))
        self.loc = tuple(loc)
        self._update_location()
        # Update the canvas size.
        self.canvas_size = event.size

    def on_mouse_move(self, event):
        # Make sure the axis is rotating with the camera.
        if event.button == 1 and event.is_dragging:
            self._update_axis()

    def set_anchor(self, mouse_press_event):
        """
        Set an anchor point (2D screen coordinates) when left click
        in the selection mode (<Ctrl> pressed). After that, the dragging called
        in func 'drag_visual_node' will try to move around the screen and let the
        anchor follows user's mouse position.
        """
        # Get click coordinates on the screen.
        click_pos = mouse_press_event.pos
        # Compute the anchor coordinates by subtracting the click_pos with
        # self.loc in order to get a relative anchor position.
        self.anchor = list(np.array(click_pos[:2]) - np.array(self.loc))

    def drag_visual_node(self, mouse_move_event):
        """
        Drag this visual node while holding left click in the selection mode
        (<Ctrl> pressed). The highlighted axis will move with the mouse (the anchor
        point will be 'hooked' to the mouse).
        """
        # Compute the new axis legend center via dragging.
        new_center = mouse_move_event.pos[:2] - self.anchor
        # Get the offset from new_center to current axis center self.loc.
        self.offset = new_center - self.loc

        if hasattr(self, 'highlight'):
            self.highlight.center = new_center  # move highlighte together with axis
        self._update_location()

    def _update_location(self):
        """
        Update the axis plane to the dragged location and redraw.
        """
        self.loc += self.offset
        self.loc = tuple(self.loc)
        self._update_axis()

        # Reset attributes after dragging completes.
        self.offset = np.array([0, 0])

    def _update_axis(self, loc=None):
        """
        Align the axis legend with the current camera rotation, scale up to
        desired size, then put the axis legend at pos (measuring from the
        top-left corner).
        """
        if self.parent is None:
            return  # if not connected to any parent, no need to update

        if loc is not None:
            self.loc = loc

        tr = self.transform
        tr.reset()  # remove all transforms (put back to origin)

        # Revert y and z axis in seismic coordinate system.
        # if self.seismic_coord_system:
        # #     tr.scale((1, -1, -1))
        tr.scale(self.flips)

        # Align camera rotation. This only works for turntable camera!!
        # for some reason I have to do this at first ...
        tr.rotate(90, (1, 0, 0))
        tr.rotate(self.parent.camera.azimuth, (0, 1, 0))
        tr.rotate(self.parent.camera.elevation, (1, 0, 0))
        # Scale up to desired size and put to desired location.
        tr.scale((*(self.size, ) * 2, 0.001))  # only scale in screen plane!
        tr.translate(self.loc)

        self.update()


class XYZAxis(scene.visuals.XYZAxis, FixMixin):
    """
    An XYZAxis legend visual that add some interactivity fixes to be used
    in 3D. It rotates correctly, and can be dragged around.

    Parameters
    ----------
    loc : Tuple
        2D position for the location of the axis
    size : int
        size of axis
    width : int
        axis line width
    antialias : bool
        antialias of XYZAxis
    visable : bool
        visable when show
    parent : Any
        parant of this None
    """

    def __init__(self,
                 loc=(30, 30),
                 size=60,
                 width=2,
                 antialias=True,
                 visible=True,
                 parent=None):
        # Create a scene.visuals.XYZAxis (without parent by default).
        scene.visuals.XYZAxis.__init__(self,
                                       parent=parent,
                                       width=width,
                                       antialias=antialias)
        self.interactive = True
        self.unfreeze()
        self.ids = 'xyz'  # for link camera move
        self.visible = visible
        self.canvas_size = None  # will be set when parent is linked

        # Determine the size and position.
        self.loc = loc
        self.size = size

        # z-axis down seismic coordinate system, or z-axis up normal system.
        xflip = -1 if cigvis.is_x_reversed() else 1
        yflip = -1 if cigvis.is_y_reversed() else 1
        zflip = -1 if cigvis.is_z_reversed() else 1
        flips = [xflip, yflip, zflip]
        self.flips = flips

        # The selection highlight (a Ellipse visual with transparent color).
        # The circle is centered on the axis legend.
        self.highlight = scene.visuals.Ellipse(
            parent=parent,
            center=self.loc,
            radius=self.size,
            color=(1, 1, 0, 0.5))  # transparent yellow color
        self.highlight.visible = False  # only show when selected

        # Set the anchor point (2D screen coordinates). The mouse will
        # drag the axis by anchor point to move around the screen.
        self.anchor = None  # None by default
        self.offset = np.array([0, 0])

        # The axis legend is rotated to align with the parent camera. Then put
        # the legend to specified location and scale up to desired size.
        # The location is computed from the top-left corner.
        self.transform = MatrixTransform()

        self.freeze()


class NorthPointer(Compound, FixMixin):

    def __init__(self,
                 direction,
                 scale=2,
                 loc=(30, 30),
                 size=60,
                 style='default'):
        vertices, faces, face_colors, text_pos = north_pointer_mesh(
            direction, scale, style)

        self._mesh = Mesh(vertices, faces, face_colors=face_colors)
        self._text = Text('N',
                          pos=text_pos,
                          color='black',
                          bold=True,
                          depth_test=True)
        Compound.__init__(self, [self._mesh, self._text])
        self.unfreeze()
        self.dyn_light = False
        # Determine the size and position.
        self.loc = loc
        self.size = size
        self.canvas_size = None  # will be set when parent is linked
        self.anchor = None  # None by default
        self.offset = np.array([0, 0])

        xflip = -1 if cigvis.is_x_reversed() else 1
        yflip = -1 if cigvis.is_y_reversed() else 1
        zflip = -1 if cigvis.is_z_reversed() else 1
        flips = [xflip, yflip, zflip]
        self.flips = flips

        self.transform = MatrixTransform()
        self.freeze()


# TODO: finished the Units
# We need a node:
# 1. auto resize when zoom in/out
# 2. never rotate
# 3. never move
class Units(Compound):

    def __init__(self, size, label):
        self._line = Line(np.array([[100, 100, 100], [size + 100, 100, 100]]),
                          width=8)
        self._text = Text(label, pos=[size / 2 + 100, 100, 100])
        Compound.__init__(self, [self._line, self._text])
        raise NotImplementedError("TODO: Need implement this")
