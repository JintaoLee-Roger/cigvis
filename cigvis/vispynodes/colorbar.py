# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2023, modified by Jintao Li.
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC)
#
#
# Copyright (C) 2019 Yunzhi Shi @ The University of Texas at Austin.
# All rights reserved.
# Distributed under the MIT License. See LICENSE for more info.
# ------------------------------------------------------------------------------

import io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from vispy import scene
import vispy.color as viscolor
from vispy.util.dpi import get_dpi
from vispy.visuals.transforms import MatrixTransform

from cigvis import colormap


class Colorbar(scene.visuals.Image):
    """
    A colorbar visual fixed to the right side of the canvas. This is
    based on the rendering from Matplotlib, then display this rendered
    image as a scene.visuals.Image visual node on the canvas.

    Parameters
    ----------
    size : Tuple
        cbar image size: (width, hight)
    cmap : Colormap
        cmap
    clim : List
        [vmin, vmax] of the cmap
    discrete : bool
        set as discrete colorbar style if True
    disc_ticks : List
        Two elements, [values, ticklabels]
        The first element is a List which contains the avilable values,
        it will be used to calculate the colors from cmap and clim, i.e,
        get: values <--> colors. The second element is a List which contains
        the labels of all values, such as 'class A', 'class B', .... If 
        `len(disc_ticks) == 1`, use values as the second element.
    dpi_scale : float
        dpi scale when drawing, default is 1.5, true_dpi = dpi * dpi_scale
    label_str : str
        Colorbar label
    label_color : str
        colorbar label color
    label_size : int
        colorbar label size
    tick_size : int
        tick size
    border_width : float
        border size
    border_color : Color
        border color
    savedir : str
        colobar image save dir
    visable : bool
        visable when show
    parent : Any
        parent of this None
    """

    def __init__(
            self,
            size=None,  # width, hight
            cmap='gray',
            clim=None,
            discrete=False,
            disc_ticks=None,
            dpi_scale=1.5,
            label_str="Colorbar",
            label_color='black',
            label_size=None,
            tick_size=None,
            border_width=None,
            border_color='black',
            savedir=None,
            visible=True,
            parent=None):

        assert clim is not None, 'clim must be specified explicitly.'

        # Create a scene.visuals.Image (without parent by default).
        scene.visuals.Image.__init__(self,
                                     parent=parent,
                                     interpolation='bicubic',
                                     method='auto')
        self.unfreeze()
        self.savedir = savedir
        self.cbar_name = None

        self.visible = visible
        self.dpi_scale = dpi_scale

        # Record the important drawing parameters.
        self.cbar_size = size  # tuple
        self.cmap_ = cmap
        self.discrete = discrete
        if self.discrete:
            assert disc_ticks is not None
        self.disc_ticks = disc_ticks
        self.clim_ = clim  # tuple

        # Record the styling parameters.
        self.label_str = label_str
        self.label_color = label_color
        self.label_size = label_size
        self.tick_size = tick_size
        self.border_width = border_width
        self.border_color = border_color

        # Draw colorbar using Matplotlib.
        if self.cbar_size is not None:
            self.set_data(self._draw_colorbar())

        # Give a Matrix transform to self in order to move around canvas.
        self.transform = MatrixTransform()

        self.freeze()

    def update_size(self, size):
        self.unfreeze()
        self.cbar_size = size
        self.set_data(self._draw_colorbar())
        self.freeze()

    def update_params(self, **kwargs):
        self.unfreeze()
        self.savedir = kwargs.get('savedir', self.savedir)
        self.cbar_name = kwargs.get('cbar_name', self.cbar_name)
        self.cbar_size = kwargs.get('cbar_size', self.cbar_size)
        self.cmap_ = kwargs.get('cmap', self.cmap_)
        self.discrete = kwargs.get('discrete', self.discrete)
        self.disc_ticks = kwargs.get('disc_ticks', self.disc_ticks)
        self.dpi_scale = kwargs.get('dpi_scale', self.dpi_scale)
        if self.discrete:
            assert self.disc_ticks is not None
        self.clim_ = kwargs.get('clim', self.clim_)
        self.label_str = kwargs.get('label_str', self.label_str)
        self.label_color = kwargs.get('label_color', self.label_color)
        self.label_size = kwargs.get('label_size', self.label_size)
        self.tick_size = kwargs.get('tick_size', self.tick_size)
        self.border_width = kwargs.get('border_width', self.border_width)
        self.border_color = kwargs.get('border_color', self.border_color)
        self.set_data(self._draw_colorbar())
        self.freeze()

    def _draw_colorbar(self):
        """
        Draw a Matplotlib colorbar, save this figure without any boundary to a
        rendering buffer, and return this buffer as a numpy array.
        """
        assert self.cbar_size is not None
        dpi = get_dpi()
        figsize = (self.cbar_size[0] / dpi,
                   self.cbar_size[1] / dpi*0.95)

        sm, ticks = self.get_ScalarMappable()

        # Put the colorbar at proper location on the Matplotlib fig.
        fig, ax = plt.subplots(figsize=figsize)
        cb = fig.colorbar(sm, cax=ax)
        if self.discrete:
            cb.set_ticks(ticks['ticks'])
            cb.set_ticklabels(ticks['labels'])

        # Apply styling to the colorbar.
        cb.set_label(
            self.label_str,
            color=self.label_color,
            size=self.label_size,
        )
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=self.label_color)
        cb.ax.yaxis.set_tick_params(
            color=self.label_color,
            labelsize=self.tick_size,
        )
        cb.outline.set_linewidth(self.border_width)
        cb.outline.set_edgecolor(self.border_color)

        plt.tight_layout()

        # Export the rendering to a numpy array in the buffer.
        buf = io.BytesIO()
        # To remove the white gap between the image and the cbar
        # set `bbox_inches='tight'`
        fig.savefig(
            buf,
            format='png',
            bbox_inches='tight',
            pad_inches=0.02,
            dpi=dpi * self.dpi_scale,
            transparent=True,
        )

        if self.cbar_name is not None:
            fig.savefig(
                Path(self.savedir) / self.cbar_name,
                format='png',
                bbox_inches='tight',
                pad_inches=0.02,
                dpi=dpi * 4,
                transparent=True,
            )

        plt.close()

        buf.seek(0)
        img = plt.imread(buf, 'png')
        buf.flush()

        return img

    def get_ScalarMappable(self):
        # To matplotlib cmap
        if isinstance(self.cmap_, viscolor.Colormap):
            rgba = self.cmap_.colors.rgba
            # Blend to white to avoid this Matplotlib rendering issue:
            # https://github.com/matplotlib/matplotlib/issues/1188
            for i in range(3):
                rgba[:, i] = (1 - rgba[:, -1]) + rgba[:, -1] * rgba[:, i]
            rgba[:, -1] = 1.
            if len(rgba) < 2:  # in special case of 'grays' cmap!
                rgba = np.array([[0, 0, 0, 1.], [1, 1, 1, 1.]])
            cmap = LinearSegmentedColormap.from_list('vispy_cmap', rgba)
        elif isinstance(self.cmap_, str):
            cmap = colormap.get_cmap_from_str(self.cmap_)
        elif isinstance(self.cmap_, mpl.colors.Colormap):
            cmap = self.cmap_
        else:
            raise RuntimeError("error type of cmap")

        if not self.discrete:
            norm = mpl.colors.Normalize(vmin=self.clim_[0], vmax=self.clim_[1])
            ticks = None
        else:
            ticks = {}
            # get the correspanding colors from cmap, and clim
            colors = colormap.get_colors_from_cmap(cmap, self.clim_,
                                                   self.disc_ticks[0])
            cmap = ListedColormap(colors)
            # norm of equal intervals
            norm = mpl.colors.BoundaryNorm(
                np.arange(0, (len(colors) + 1) * 2, 2), cmap.N)
            # set ticks in the center of boundaries
            ticks['ticks'] = np.arange(1, len(colors) * 2, 2)
            ticks['labels'] = self.disc_ticks[0]
            if len(self.disc_ticks) == 2:
                assert len(self.disc_ticks[0]) == len(self.disc_ticks[1])
                ticks['labels'] = self.disc_ticks[1]

        return plt.cm.ScalarMappable(cmap=cmap, norm=norm), ticks
