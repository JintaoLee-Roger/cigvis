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
from typing import List
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
            dpi_scale=4,
            label_str="Colorbar",
            label_color='black',
            label_size=None,
            label_bold=False,
            tick_size=None,
            border_width=None,
            border_color='black',

            height_ratio=0.96,
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
        self.pos = (0, 0)
        self.canvas_size = None
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
        self.label_bold = 'bold' if label_bold else 'normal'
        assert height_ratio > 0 and height_ratio <= 1
        self.height_ratio = height_ratio

        if self.label_size is None:
            self.label_size = plt.rcParams['axes.labelsize']
        if self.tick_size is None:
            self.tick_size = plt.rcParams['ytick.labelsize']
        if self.border_width is None:
            self.border_width = plt.rcParams['lines.linewidth']
        self.tick_length = plt.rcParams['ytick.major.size']
        self.tick_width = plt.rcParams['ytick.major.width']

        [self.label_size, self.tick_size] = self.get_font_size([self.label_size, self.tick_size]) # yapf: disable

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

    def on_resize(self, event):
        """ When window is resized, only need to move the position in vertical
        direction, because the coordinate is relative to the secondary ViewBox
        that stays on the right side of the canvas.
        """
        pos = np.array(self.pos).astype(np.float32)
        pos[1] *= event.size[1] / self.canvas_size[1]
        self.pos = tuple(pos)

        scale = 1 / self.dpi_scale * self.height_ratio

        # Move the colorbar to specified position (with half-size padding, because
        # Image visual uses a different anchor (top-left corner) rather than the
        # center-left corner used by ColorBar visual.).
        self.transform.reset()
        self.transform.translate((
            1/scale,  # Keep the colorbar close to the left side 
            # self.pos[1] * self.dpi_scale * 0.05,
            (self.pos[1] - self.size[1] / 2. * scale) / scale,  # Move the colorbar to the center
        ))
        self.transform.scale([scale, scale])

        # Update the canvas size.
        self.canvas_size = event.size

    def _draw_colorbar(self):
        """
        Draw a Matplotlib colorbar, save this figure without any boundary to a
        rendering buffer, and return this buffer as a numpy array.
        """
        assert self.cbar_size is not None

        # Using `self.dpi_scale`` to zoom in the font, produce a high resolution colorbar
        dpi = get_dpi()
        figsize = (self.cbar_size[0] / dpi * self.dpi_scale,
                   self.cbar_size[1] / dpi * self.dpi_scale)

        sm, ticks = self.get_ScalarMappable()

        # Put the colorbar at proper location on the Matplotlib fig.
        fig = plt.figure(figsize=figsize)
        width = figsize[1] * 0.2 / figsize[0] / 5
        cbar_axes = fig.add_axes([0.01, 0.01, width, 0.98])
        cb = fig.colorbar(sm, cax=cbar_axes)
        if self.discrete:
            cb.set_ticks(ticks['ticks'])
            cb.set_ticklabels(ticks['labels'])

        # Apply styling to the colorbar.
        cb.set_label(
            self.label_str,
            color=self.label_color,
            fontsize=self.label_size * self.dpi_scale,
            fontweight=self.label_bold,
        )
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=self.label_color)
        cb.ax.yaxis.set_tick_params(
            color=self.label_color,
            labelsize=self.tick_size * self.dpi_scale,
            length=self.tick_length * self.dpi_scale,
            width=self.tick_width * self.dpi_scale,
        )
        cb.ax.yaxis.get_offset_text().set_size(self.tick_size * self.dpi_scale)
        cb.outline.set_linewidth(self.border_width * self.dpi_scale)
        cb.outline.set_edgecolor(self.border_color)

        # Export the rendering to a numpy array in the buffer.
        buf = io.BytesIO()
        # To remove the white gap between the image and the cbar
        # set `bbox_inches='tight'`
        fig.savefig(
            buf,
            format='png',
            bbox_inches='tight',
            pad_inches=0.04,
            dpi=dpi,
            transparent=True,
        )

        if self.cbar_name is not None:
            fig.savefig(
                Path(self.savedir) / self.cbar_name,
                format='png',
                bbox_inches='tight',
                pad_inches=0.04,
                dpi=dpi,
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

    def get_font_size(self, fonts):
        if not isinstance(fonts, List):
            fontsl = [fonts]
        else:
            fontsl = fonts
        fig, ax = plt.subplots()
        t = ax.text(0.5, 0.5, 'Text')
        font_size = []
        for font in fontsl:
            if isinstance(font, (int, float)):
                font_size.append(font)
            else:
                t.set_fontsize(font)
                font_size.append(round(t.get_fontsize(), 2))
        plt.close(fig)
        if not isinstance(fonts, List):
            font_size = font_size[0]
        return font_size
