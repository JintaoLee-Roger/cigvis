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

from typing import Callable, List
import numpy as np
from vispy.scene.visuals import Image, Line, Plane
from vispy.visuals.transforms import MatrixTransform, STTransform
from vispy.gloo.wrappers import set_polygon_offset
import cigvis


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

        # Get the image_func that returns either image or image shape.
        self.image_funcs = image_funcs  # a list of functions!
        shape = self.image_funcs[0](self.pos, get_shape=True)
        # self._shape = shape

        # The selection highlight (a Plane visual with transparent color).
        # The plane is initialized before any rotation, on '+z' direction.
        self.highlight = Plane(
            parent=self,
            width=shape[0],
            height=shape[1],
            direction='+z',
            color=(1, 1, 0, 0.1),  # transparent yellow color
        )
        # Move the plane to align with the image.
        self.highlight.transform = STTransform(translate=(shape[0] / 2,
                                                          shape[1] / 2, 0))
        # This is to make sure we can see highlight plane through the images.
        self.highlight.set_gl_state('additive', depth_test=True)
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

    def set_visable(self, idx: int, visable=False):
        if idx <= 0:
            return
        self.overlaid_images[idx].visible = visable

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
        if self.axis == 'x':
            return [self.pos, *pos]
        if self.axis == 'y':
            return [pos[0], self.pos, pos[1]]
        if self.axis == 'z':
            return [*pos, self.pos]

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
        if self.axis == 'z':
            # 1. No rotation to do for z axis (y-x) slice. Only translate.
            self.transform.translate((0, 0, pos2))
        elif self.axis == 'y':
            # 2. Rotation(s) for the y axis (z-x) slice, then translate:
            self.transform.rotate(90, (1, 0, 0))
            self.transform.translate((0, pos2, 0))
        elif self.axis == 'x':
            # 3. Rotation(s) for the x axis (z-y) slice, then translate:
            self.transform.rotate(90, (1, 0, 0))
            self.transform.rotate(90, (0, 0, 1))
            self.transform.translate((pos2, 0, 0))

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
        # Note: self.size[0] is slow dim size, self.size[1] is fast dim size.
        if self.axis == 'z':
            if axis_3d == 0: return (0, self.size[0])
            elif axis_3d == 1: return (0, self.size[1])
            elif axis_3d == 2: return (self.pos, self.pos)
        elif self.axis == 'y':
            if axis_3d == 0: return (0, self.size[0])
            elif axis_3d == 1: return (self.pos, self.pos)
            elif axis_3d == 2: return (0, self.size[1])
        elif self.axis == 'x':
            if axis_3d == 0: return (self.pos, self.pos)
            elif axis_3d == 1: return (0, self.size[0])
            elif axis_3d == 2: return (0, self.size[1])

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
    ):
        super().__init__(pos, color, width, connect, method, antialias)
        """
        """
        self.unfreeze()
        self.axis_pair = axis_pair
        self.shape = shape
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
        # fmt: off
        if axis == 'x':
            if pos == self.shape[0] - 1:
                pos += 1
            lines = [[pos, 0, 0], [pos, 0, self.shape[2]], [pos, self.shape[1], self.shape[2]], [pos, self.shape[1], 0], [pos, 0, 0]]
        elif axis == 'y':
            if pos == self.shape[1] - 1:
                pos += 1
            lines = [[0, pos, 0], [0, pos, self.shape[2]], [self.shape[0], pos, self.shape[2]], [self.shape[0], pos, 0], [0, pos, 0]]
        else:
            if pos == self.shape[2] - 1:
                pos += 1
            lines = [[0, 0, pos], [0, self.shape[1], pos], [self.shape[0], self.shape[1], pos], [self.shape[0], 0, pos], [0, 0, pos]]
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
        if pos_a == self.shape[a_idx] - 1:
            pos_a += 1
        if pos_b == self.shape[b_idx] - 1:
            pos_b += 1

        start = [0] * 3
        start[a_idx] = pos_a
        start[b_idx] = pos_b

        end = list(start)
        end[third_axis] = self.shape[third_axis]

        self.set_data(np.array([start, end]))


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
    def _eq_3_or_4(k):
        return k == 3 or k == 4
    line_first = cigvis.is_line_first()
    assert _eq_3_or_4(vol.ndim), f"Volume's dims must be 3 or 4 (RGB), but got {vol.ndim}"
    # rgb_type, 0 for (n1, n2, n3), 1 for (n1, n2, n3, 3/4), 2 for (3/4, n1, n2, n3)
    ndim = vol.ndim
    channel_dim = None
    dim_x, dim_y, dim_z = (0, 1, 2) if line_first else (2, 1, 0)
    axis_to_dim = {'x': dim_x, 'y': dim_y, 'z': dim_z}
    shape, rgb_type = cigvis.utils.get_shape(vol, line_first)
    if rgb_type == 1:
        channel_dim = 3
    elif rgb_type == 2:
        channel_dim = 0

    def wrap_preproc_f(x, func=None, forcefp32=False):
        if line_first and rgb_type == 1:
            x = np.transpose(x, (1, 2, 0))
        elif (not line_first) and rgb_type == 2:
            x = np.transpose(x, (1, 2, 0))
        if func is not None:
            x = func(x)
        if not forcefp32:
            return x

        x = np.array(x)
        if x.dtype == np.float16:
            x = x.astype(np.float32)
        return x

    _preproc_f = lambda x: wrap_preproc_f(x, preproc_f, forcefp32)

    def _get_slices(axis, pos):
        dim = axis_to_dim.get(axis)
        slices = [slice(None)] * ndim
        if channel_dim is not None and dim >= channel_dim:
            slices[dim + 1] = pos
        else:
            slices[dim] = pos
        return tuple(slices)

    def slicing_at_axis(pos, get_shape=False):
        if get_shape:  # just return the shape information
            if axis == 'x': return shape[1], shape[2]
            elif axis == 'y': return shape[0], shape[2]
            elif axis == 'z': return shape[0], shape[1]
        else:  # will slice the volume and return an np array image
            pos = int(np.round(pos))
            s = _get_slices(axis, pos)
            if line_first:
                return _preproc_f(vol[s].T)
            else:
                return _preproc_f(vol[s])
            # if line_first:
            #     if axis == 'x': 
            #         if rgb_type == 2:
            #             return _preproc_f(vol[:, pos, :, :].T)
            #         else:
            #             return _preproc_f(vol[pos, :, :].T)
            #     elif axis == 'y': return _preproc_f(vol[:, pos, :].T)
            #     elif axis == 'z': return _preproc_f(vol[:, :, pos].T)
            # else:
            #     if axis == 'x': return _preproc_f(vol[:, :, pos])
            #     elif axis == 'y': return _preproc_f(vol[:, pos, :])
            #     elif axis == 'z': return _preproc_f(vol[pos, :, :])

    return slicing_at_axis
