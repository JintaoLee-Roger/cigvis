# Copyright (c) 2023 Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).
# All rights reserved.

from typing import List, Tuple
import numpy as np
from types import SimpleNamespace
from vispy import scene
from vispy.visuals.transforms import MatrixTransform
from cigvis import is_line_first


class FixedImage(scene.visuals.Image):
    """
    fixed location image, input slices, instead of volume, disable
    drag mode

    Parameters
    -----------
    imgs : array-like or List[array-like]
        input images
    axis : str
        one of ['x', 'y', 'z']
    pos : int or List or Tuple
        position of the image, can be int number or Tuple,
        pos=10, axis='z' => start_pos = [0, 0, 10],
        pos=[90, 10, 20] => start_pos = [90, 10, 20]
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
                 imgs,
                 axis='z',
                 pos=0,
                 cmaps=['grays'],
                 clims=None,
                 interpolations=['linear'],
                 method='auto'):
        assert clims is not None, 'clim must be specified explicitly.'
        if not isinstance(imgs, List):
            imgs = [imgs]
        if not isinstance(cmaps, List):
            cmaps = [cmaps]
        if not isinstance(interpolations, List):
            interpolations = [interpolations]
        if not isinstance(clims[0], List):
            clims = [clims]
        if is_line_first():
            imgs = [img.T for img in imgs]

        scene.visuals.Image.__init__(
            self,
            parent=None,  # no image func yet
            cmap=cmaps[0],
            clim=clims[0],
            interpolation=interpolations[0],
            method=method)

        self.unfreeze()
        self.imgs = imgs
        self.interactive = True

        # Other images ...
        self.overlaid_images = [self]
        for i_img in range(1, len(imgs)):
            overlaid_image = scene.visuals.Image(
                parent=self,
                cmap=cmaps[i_img],
                clim=clims[i_img],
                interpolation=interpolations[i_img],
                method=method)
            self.overlaid_images.append(overlaid_image)

        # Set GL state. Must check depth test, otherwise weird in 3D.
        self.set_gl_state(depth_test=True,
                          blend=True,
                          depth_func='lequal',
                          blend_func=('src_alpha', 'one_minus_src_alpha'))

        assert axis in ['x', 'y', 'z']
        self.axis = axis

        if isinstance(pos, (int, np.integer)):
            if axis == 'x':
                self.pos = (pos, 0, 0)
            elif axis == 'y':
                self.pos = (0, pos, 0)
            else:
                self.pos = (0, 0, pos)
        elif isinstance(pos, (List, Tuple)):
            assert len(pos) == 3
            self.pos = pos
        else:
            raise TypeError("pos must be a int or 3 elements List/Tuple")

        self.highlight = SimpleNamespace()
        self.highlight.visable = False

        # Apply SRT transform according to the axis attribute.
        self.transform = MatrixTransform()
        # Move the image plane to the corresponding location.
        self._update_location()

        self.freeze()

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

    def _update_location(self):
        # self.pos = int(np.round(self.pos))

        # Update the transformation in order to move to new location.
        self.transform.reset()
        if self.axis == 'z':
            # 1. No rotation to do for z axis (y-x) slice. Only translate.
            # self.transform.translate((0, 0, self.pos))
            self.transform.translate(self.pos)
        elif self.axis == 'y':
            # 2. Rotation(s) for the y axis (z-x) slice, then translate:
            self.transform.rotate(90, (1, 0, 0))
            # self.transform.translate((0, self.pos, 0))
            self.transform.translate(self.pos)
        elif self.axis == 'x':
            # 3. Rotation(s) for the x axis (z-y) slice, then translate:
            self.transform.rotate(90, (1, 0, 0))
            self.transform.rotate(90, (0, 0, 1))
            # self.transform.translate((self.pos, 0, 0))
            self.transform.translate(self.pos)

        self.set_data(self.imgs[0])  # remove .T
        # Other images, overlaid on the primary image:
        for i_img in range(1, len(self.imgs)):
            self.overlaid_images[i_img].set_data(self.imgs[i_img])

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
            elif axis_3d == 2: return (self.pos[2], self.pos[2])
        elif self.axis == 'y':
            if axis_3d == 0: return (0, self.size[0])
            elif axis_3d == 1: return (self.pos[1], self.pos[1])
            elif axis_3d == 2: return (0, self.size[1])
        elif self.axis == 'x':
            if axis_3d == 0: return (self.pos[0], self.pos[0])
            elif axis_3d == 1: return (0, self.size[0])
            elif axis_3d == 2: return (0, self.size[1])
