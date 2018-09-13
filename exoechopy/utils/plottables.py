"""
This module provides base classes for objects that can be plotted with the visualize tools.
"""

import numpy as np
from collections import Sequence
from .globals import mpl_colors

__all__ = ['Plottable', 'PointCloud']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class Plottable:
    """Mixin that describes how to plot an object."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 point_color=None,
                 point_size=None,
                 path_color=None,
                 linewidth=None,
                 name=None,
                 display_marker=None,
                 **kwargs):

        super().__init__(**kwargs)

        if point_color is None:
            self._point_color = 'k'
        else:
            self._point_color = point_color

        if point_size is None:
            self._point_size = 20
        else:
            self._point_size = point_size

        if path_color is None:
            self._path_color = 'gray'
        else:
            self._path_color = path_color

        if linewidth is None:
            self._linewidth = 1.
        else:
            self._linewidth = linewidth

        if name is None:
            self._name = ""
        else:
            self._name = name

        if display_marker is None:
            self._display_marker = 'o'
        else:
            self._display_marker = display_marker

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def path_color(self):
        return self._path_color

    @path_color.setter
    def path_color(self, path_color):
        self._path_color = path_color

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def point_color(self):
        return self._point_color

    @point_color.setter
    def point_color(self, point_color):
        self._point_color = point_color

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, linewidth):
        self._linewidth = linewidth

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def point_size(self):
        return self._point_size

    @point_size.setter
    def point_size(self, point_size):
        self._point_size = point_size

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def display_marker(self):
        return self._display_marker

    @display_marker.setter
    def display_marker(self, display_marker):
        self._display_marker = display_marker




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class PointCloud(Plottable, Sequence):
    def __init__(self,
                 point_array: np.array,
                 **kwargs):
        """
        A class to help plot point clouds from an array of (x, y, z) points
        :param point_array:
        """
        super().__init__(**kwargs)
        self._point_array = np.transpose(point_array)

    def __getitem__(self, i: int):
        return self._point_array[i]

    def __len__(self):
        return len(self._point_array)
