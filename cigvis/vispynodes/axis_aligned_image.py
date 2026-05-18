# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, modified by Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC)
#
# Copyright (C) 2019 Yunzhi Shi @ The University of Texas at Austin.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for more info.
# -----------------------------------------------------------------------------

from typing import Callable, Dict, List, Tuple
import numpy as np
from vispy.scene.visuals import Image, Line, Plane
from vispy.visuals.transforms import MatrixTransform, STTransform
from vispy.gloo.wrappers import set_polygon_offset
import cigvis
from cigvis.utils.slice_provider import SliceProvider


class AxisAlignedImage(Image):
    """
    Visual subclass displaying an image that aligns to an axis.
    This image should be able to move along the perpendicular direction when
    user gives corresponding inputs.

    Parameters
    ----------
    image_func : List[array-like]
        input images
    axis : str
        one of ['x', 'y', 'z']
    pos : int
        position of the image, can be int number or Tuple,
        pos=10, axis='z' => start_pos = [0, 0, 10]
    limit : Tuple
        limit of the viewbox
    cmaps : List
        cmap for each image
    clims : List
        clim for each image
    interpolations : List
        interpolation method for each image
    method : str
        method of `Image`
    """

    def __init__(self,
                 image_funcs: List[Callable],
                 axis='z',
                 pos=0,
                 limit=None,
                 cmaps=['grays'],
                 clims=None,
                 interpolation=['linear'],
                 method='auto',
                 texture_format=None,
                 display_range: Dict[str, Tuple[int, int]] = None,
                 offset_factor=2.0,
                 offset_units=2.0):

        assert clims is not None, 'clim must be specified explicitly.'

        # Create an Image obj and unfreeze it so we can add more
        # attributes inside.
        # First image (from image_funcs[0])
        Image.__init__(
            self,
            parent=None,  # no image func yet
            cmap=cmaps[0],
            clim=clims[0],
            interpolation=interpolation[0],
            method=method,
            texture_format=texture_format,
        )
        self.unfreeze()

        self.ids = f'{axis}{pos}'

        self.interactive = True

        self._offset_factor = offset_factor
        self._offset_units = offset_units

        # lines
        self._observers = []  # observer the intersection lines

        # Other images ...
        self.overlaid_images = [self]
        for i_img in range(1, len(image_funcs)):
            overlaid_image = Image(
                parent=self,
                cmap=cmaps[i_img],
                clim=clims[i_img],
                interpolation=interpolation[i_img],
                method=method,
                texture_format=texture_format,
            )
            self.overlaid_images.append(overlaid_image)

        # Set GL state. Must check depth test, otherwise weird in 3D.
        self.set_gl_state(depth_test=True,
                          blend=True,
                          depth_func='lequal',
                          blend_func=('src_alpha', 'one_minus_src_alpha'))

        # Determine the axis and position of this plane.
        self.axis = axis
        # Check if pos is within the range.
        if limit is not None:
            assert (pos>=limit[0]) and (pos<=limit[1]), \
              'pos={} is outside limit={} range.'.format(pos, limit)
        self.pos = pos
        self.limit = limit
        self.display_range = display_range

        # Get the image_func that returns either image or image shape.
        self.image_funcs = image_funcs  # a list of functions!
        shape = self.image_funcs[0](self.pos, get_shape=True)
        self._image_size = self._image_local_size(shape)
        highlight_width, highlight_height = self._image_size

        # The selection highlight (a Plane visual with transparent color).
        # The plane is initialized before any rotation, on '+z' direction.
        self.highlight = Plane(
            parent=self,
            width=highlight_width,
            height=highlight_height,
            direction='+z',
            color=(1, 1, 0, 0.1),  # transparent yellow color
        )
        # Move the plane to align with the image.
        self.highlight.transform = STTransform(
            translate=(highlight_width / 2, highlight_height / 2, 0))
        self.highlight.set_gl_state(
            depth_test=True,
            blend=True,
            depth_func='lequal',
            blend_func=('src_alpha', 'one_minus_src_alpha'),
        )
        self.highlight.visible = False  # only show when selected

        # Set the anchor point (2D local world coordinates). The mouse will
        # drag this image by anchor point moving in the normal direction.
        self.anchor = None  # None by default
        self.offset = 0

        # Apply SRT transform according to the axis attribute.
        self.transform = MatrixTransform()
        # Move the image plane to the corresponding location.
        self._update_location()

        self.freeze()

    def _image_local_size(self, shape):
        # vispy Image maps array shape (rows, cols[, channels]) to local
        # coordinates (width=cols, height=rows).
        return int(shape[1]), int(shape[0])

    def add_mask(self,
                 vol: np.ndarray,
                 cmap: str,
                 clim: List,
                 interpolation: str,
                 method: str = 'auto',
                 texture_format: str = 'auto',
                 preproc_f: Callable = None):
        self.unfreeze()
        image_func = get_image_func(self.axis, vol, preproc_f, True)
        self.image_funcs.append(image_func)

        self.overlaid_images.append(
            Image(
                parent=self,
                cmap=cmap,
                clim=clim,
                interpolation=interpolation,
                method=method,
                texture_format=texture_format,
            ))
        self._update_location()
        self.freeze()

    def remove_mask(self, idx):
        if idx <= 0:
            return
        self.unfreeze()

        image = self.overlaid_images.pop(idx)
        image.parent = None
        del image
        image_func = self.image_funcs.pop(idx)
        del image_func

        # self._update_location()
        self.freeze()

    def set_visible(self, idx: int, visible=False):
        if idx <= 0:
            return
        self.overlaid_images[idx].visible = visible

    @property
    def axis(self):
        """
        The dimension that this image is perpendicular aligned to.
        """
        return self._axis

    @axis.setter
    def axis(self, value):
        value = value.lower()
        if value not in ('z', 'y', 'x'):
            raise ValueError('Invalid value for axis.')
        self._axis = value

    def get_click_pos3d(self, mouse_press_event):
        pos = self._click_pos(mouse_press_event)
        x0, _ = self._display_axis_range('x')
        y0, _ = self._display_axis_range('y')
        z0, _ = self._display_axis_range('z')
        if self.axis == 'x':
            return [self.pos, y0 + pos[0], z0 + pos[1]]
        if self.axis == 'y':
            return [x0 + pos[0], self.pos, z0 + pos[1]]
        if self.axis == 'z':
            return [x0 + pos[0], y0 + pos[1], self.pos]

    def _click_pos(self, mouse_press_event):
        # Get the screen-to-local transform to get camera coordinates.
        tr = self.canvas.scene.node_transform(self)

        # Get click (camera) coordinate in the local world.
        click_pos = tr.map([*mouse_press_event.pos, 0, 1])
        click_pos /= click_pos[3]  # rescale to cancel out the pos.w factor
        # Get the view direction (camera-to-target) vector in the local world.
        view_vector = tr.map([*mouse_press_event.pos, 1, 1])[:3]
        view_vector /= np.linalg.norm(view_vector)  # normalize to unit vector

        # Get distance from camera to the drag anchor point on the image plane.
        # Eq 1: click_pos + distance * view_vector = anchor
        # Eq 2: anchor[2] = 0 <- intersects with the plane
        # The following equation can be derived by Eq 1 and Eq 2.
        distance = (0. - click_pos[2]) / view_vector[2]
        # only need vec2
        pos = click_pos[:2] + distance * view_vector[:2]
        if pos[0] > self.size[0] - 1:
            pos[0] = self.size[0] - 1
        if pos[1] > self.size[1] - 1:
            pos[1] = self.size[1] - 1
        pos = [round(float(pos[0]), 4), round(float(pos[1]), 4)]
        return pos

    def set_anchor(self, mouse_press_event):
        """
        Set an anchor point (2D coordinate on the image plane) when left click
        in the selection mode (<Ctrl> pressed). After that, the dragging called
        in func 'drag_visual_node' will try to move along the normal direction
        and let the anchor follows user's mouse position as close as possible.
        """
        self.anchor = self._click_pos(mouse_press_event)

    def drag_visual_node(self, mouse_move_event):
        """
        Drag this visual node while holding left click in the selection mode
        (<Ctrl> pressed). The plane will move in the normal direction
        perpendicular to this image, and the anchor point (set with func
        'set_anchor') will move along the normal direction to stay as close to
        the mouse as possible, so that user feels like 'dragging' the plane.
        """
        # Get the screen-to-local transform to get camera coordinates.
        tr = self.canvas.scene.node_transform(self)

        # Unlike in 'set_anchor', we now convert most coordinates to the screen
        # coordinate system, because it's more intuitive for user to do operations
        # in 2D and get 2D feedbacks, e.g. mouse leading the anchor point.
        anchor = [*self.anchor, self.pos, 1]  # 2D -> 3D
        # screen coordinates of the anchor point
        anchor_screen = tr.imap(anchor)
        anchor_screen /= anchor_screen[3]  # rescale to cancel out 'w' term
        anchor_screen = anchor_screen[:2]  # only need vec2

        # Compute the normal vector, starting from the anchor point and
        # perpendicular to the image plane.
        normal = [*self.anchor, self.pos + 1, 1]  # +[0,0,1,0] from anchor
        # screen coordinates of anchor + [0,0,1,0]
        normal_screen = tr.imap(normal)
        normal_screen /= normal_screen[3]  # rescale to cancel out 'w' term
        normal_screen = normal_screen[:2]  # only need vec2
        normal_screen -= anchor_screen  # end - start = vector
        # normalize to unit vector
        if not (normal_screen[0] == 0 and normal_screen[1] == 0):
            normal_screen /= np.linalg.norm(normal_screen)

        # Use the vector {anchor_screen -> mouse.pos} and project to the
        # normal_screen direction using dot product, we can get how far the plane
        # should be moved (on the screen!).
        drag_vector = mouse_move_event.pos[:2] - anchor_screen
        # normal_screen must be length 1
        drag = np.dot(drag_vector, normal_screen)

        # We now need to convert the move distance from screen coordinates to
        # local world coordinates. First, find where the anchor is on the screen
        # after dragging; then, convert that screen point to a local line shooting
        # across the normal vector; finally, find where the line comes directly
        # above/below the anchor point (before dragging) and get that distance as
        # the true dragging distance in local coordinates.
        new_anchor_screen = anchor_screen + normal_screen * drag
        new_anchor = tr.map([*new_anchor_screen, 0, 1])
        new_anchor /= new_anchor[3]  # rescale to cancel out the pos.w factor
        view_vector = tr.map([*new_anchor_screen, 1, 1])[:3]
        view_vector /= np.linalg.norm(view_vector)  # normalize to unit vector
        # Solve this equation:
        # new_anchor := new_anchor + view_vector * ?,
        # ^^^ describe a 3D line of possible new anchor positions
        # arg min (?) |new_anchor[:2] - anchor[:2]|
        # ^^^ find a point on that 3D line that minimize the 2D distance between
        #     new_anchor and anchor.
        numerator = anchor[:2] - new_anchor[:2]
        numerator *= view_vector[:2]  # element-wise multiplication
        numerator = np.sum(numerator)
        denominator = view_vector[0]**2 + view_vector[1]**2
        shoot_distance = numerator / denominator
        # Shoot from new_anchor to get the new intersect point. The z- coordinate
        # of this point will be our dragging offset.
        offset = new_anchor[2] + view_vector[2] * shoot_distance

        # Note: must reverse normal direction from -y direction to +y!
        if self.axis == 'y': offset = -offset
        # Limit the dragging within range.
        if self.limit is not None:
            if self.pos + offset < self.limit[0]:
                offset = self.limit[0] - self.pos
            if self.pos + offset > self.limit[1]:
                offset = self.limit[1] - self.pos
        self.offset = offset
        # Note: must reverse normal direction from +y direction to -y!
        if self.axis == 'y': offset = -offset

        self._update_location()

    def _update_location(self, pos=None):
        """
        Update the image plane to the dragged location and redraw this image.
        """
        if pos is None:
            self.pos += self.offset
            # must round to nearest integer location
            self.pos = int(np.round(self.pos))
        else:
            self.pos = pos
            if self.pos < self.limit[0]:
                self.pos = self.limit[0]
            if self.pos > self.limit[1]:
                self.pos = self.limit[1]

        pos2 = self.pos
        if pos2 == self.limit[1]:
            pos2 = self.limit[1] + 1

        # Update the transformation in order to move to new location.
        self.transform.reset()
        x0, _ = self._display_axis_range('x')
        y0, _ = self._display_axis_range('y')
        z0, _ = self._display_axis_range('z')
        if self.axis == 'z':
            # 1. No rotation to do for z axis (y-x) slice. Only translate.
            self.transform.translate((x0, y0, pos2))
        elif self.axis == 'y':
            # 2. Rotation(s) for the y axis (z-x) slice, then translate:
            self.transform.rotate(90, (1, 0, 0))
            self.transform.translate((x0, pos2, z0))
        elif self.axis == 'x':
            # 3. Rotation(s) for the x axis (z-y) slice, then translate:
            self.transform.rotate(90, (1, 0, 0))
            self.transform.rotate(90, (0, 0, 1))
            self.transform.translate((pos2, y0, z0))

        # Update image on the slice based on current position. The numpy array
        # is transposed due to a conversion from i-j to x-y axis system.
        # First image, the primary one:
        self.set_data(self.image_funcs[0](self.pos))
        # Other images, overlaid on the primary image:
        for i_img in range(1, len(self.image_funcs)):
            self.overlaid_images[i_img].set_data(self.image_funcs[i_img](self.pos)) # yapf: disable

        self.update_lines()
        # Reset attributes after dragging completes.
        self.offset = 0
        self._bounds_changed()  # update the bounds with new self.pos

    def _compute_bounds(self, axis_3d, view):
        """
        Overwrite the original 2D bounds of the Image class. This will correct 
        the automatic range setting for the camera in the scene canvas. In the
        original Image class, the code assumes that the image always lies in x-y
        plane; here we generalize that to x-z and y-z plane.
        
        Parameters
        ----------
        axis_3d: int in {0, 1, 2}, represents the axis in 3D view box.
        view: the ViewBox object that connects to the parent.

        The function returns a tuple (low_bounds, high_bounds) that represents
        the spatial limits of self obj in the 3D scene.
        """
        x_range = self._display_axis_range('x')
        y_range = self._display_axis_range('y')
        z_range = self._display_axis_range('z')
        # Note: self.size[0] is slow dim size, self.size[1] is fast dim size.
        if self.axis == 'z':
            if axis_3d == 0: return x_range
            elif axis_3d == 1: return y_range
            elif axis_3d == 2: return (self.pos, self.pos)
        elif self.axis == 'y':
            if axis_3d == 0: return x_range
            elif axis_3d == 1: return (self.pos, self.pos)
            elif axis_3d == 2: return z_range
        elif self.axis == 'x':
            if axis_3d == 0: return (self.pos, self.pos)
            elif axis_3d == 1: return y_range
            elif axis_3d == 2: return z_range

    def _display_axis_range(self, axis: str):
        if self.display_range is not None and axis in self.display_range:
            return self.display_range[axis]
        if axis == self.axis:
            if self.limit is not None:
                return (self.limit[0], self.limit[1] + 1)
            return (0, self.pos + 1)
        width, height = self._image_size
        if self.axis == 'z':
            return (0, width) if axis == 'x' else (0, height)
        if self.axis == 'y':
            return (0, width) if axis == 'x' else (0, height)
        return (0, width) if axis == 'y' else (0, height)

    def _set_clipper(self, node, clipper):
        """
        To clipper its children

        Assign a clipper that is inherited from a parent node.

        If *clipper* is None, then remove any clippers for *node*.
        """
        super()._set_clipper(node, clipper)

        for im in self.children:
            if isinstance(im, Image):
                if node in im._clippers:
                    im.detach(self._clippers.pop(node))
                if clipper is not None:
                    im.attach(clipper)
                    im._clippers[node] = clipper

    def _prepare_draw(self, view):
        """
        set offet to facilitate the superimposition of lines on the image 
        """
        super()._prepare_draw(view)
        self.update_gl_state(polygon_offset_fill=True)
        set_polygon_offset(self._offset_factor, self._offset_units)

    def update_lines(self):
        for line in self._observers:
            line.refresh()

    def add_observer(self, line):
        self._observers.append(line)


class InteractiveLine(Line):

    def __init__(
        self,
        axis_pair,
        shape,
        pos=None,
        color=(1, 1, 1),
        width=1,
        connect='strip',
        method='gl',
        antialias=False,
        display_range=None,
    ):
        super().__init__(pos, color, width, connect, method, antialias)
        """
        """
        self.unfreeze()
        self.axis_pair = axis_pair
        self.shape = shape
        self.display_range = display_range
        self._linked_images = {}  # {axis: Image}
        self.freeze()

    def link_image(self, image):
        if image.axis not in self.axis_pair:
            raise ValueError("Image axis does not match line type")
        self._linked_images[image.axis] = image
        image.add_observer(self)

    def refresh(self):
        if len(self._linked_images.keys()) == 2:
            self._refresh2()
        else:
            self._refresh1()

    def _refresh1(self):
        """ update image border """
        axis = self.axis_pair[0]
        pos = self._linked_images[axis].pos
        xr = self._display_axis_range('x')
        yr = self._display_axis_range('y')
        zr = self._display_axis_range('z')
        # fmt: off
        if axis == 'x':
            pos = self._line_pos(axis, pos)
            lines = [[pos, yr[0], zr[0]], [pos, yr[0], zr[1]], [pos, yr[1], zr[1]], [pos, yr[1], zr[0]], [pos, yr[0], zr[0]]]
        elif axis == 'y':
            pos = self._line_pos(axis, pos)
            lines = [[xr[0], pos, zr[0]], [xr[0], pos, zr[1]], [xr[1], pos, zr[1]], [xr[1], pos, zr[0]], [xr[0], pos, zr[0]]]
        else:
            pos = self._line_pos(axis, pos)
            lines = [[xr[0], yr[0], pos], [xr[0], yr[1], pos], [xr[1], yr[1], pos], [xr[1], yr[0], pos], [xr[0], yr[0], pos]]
        # fmt: on
        self.set_data(np.array(lines))

    def _refresh2(self):
        """ update intersection line """
        # obtain the position of the two images
        axis_a, axis_b = self.axis_pair
        pos_a = self._linked_images[axis_a].pos if axis_a in self._linked_images else 0 # yapf: disable
        pos_b = self._linked_images[axis_b].pos if axis_b in self._linked_images else 0 # yapf: disable

        axis_order = {'x': 0, 'y': 1, 'z': 2}
        a_idx = axis_order[self.axis_pair[0]]
        b_idx = axis_order[self.axis_pair[1]]
        third_axis = 3 - a_idx - b_idx
        pos_a = self._line_pos(axis_a, pos_a)
        pos_b = self._line_pos(axis_b, pos_b)

        ranges = [
            self._display_axis_range('x'),
            self._display_axis_range('y'),
            self._display_axis_range('z'),
        ]
        start = [r[0] for r in ranges]
        start[a_idx] = pos_a
        start[b_idx] = pos_b

        end = list(start)
        end[third_axis] = ranges[third_axis][1]

        self.set_data(np.array([start, end]))

    def _display_axis_range(self, axis: str):
        if self.display_range is not None and axis in self.display_range:
            return self.display_range[axis]
        idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        return (0, self.shape[idx])

    def _line_pos(self, axis: str, pos):
        start, stop = self._display_axis_range(axis)
        if pos == stop - 1:
            return stop
        return pos


def get_image_func(axis: str,
                   vol: np.ndarray,
                   preproc_f: Callable,
                   forcefp32=False) -> Callable:
    """
    Parameters
    ----------
    axis : str
        'x' or 'y' or 'z'
    i_vol : int
        index of the volumes
    """
    provider = SliceProvider(
        vol,
        preproc=preproc_f,
        forcefp32=forcefp32,
        transpose_line_first=True,
        transpose_rgb=True,
    )

    def slicing_at_axis(pos, get_shape=False):
        return provider(axis, pos, get_shape=get_shape)

    return slicing_at_axis
