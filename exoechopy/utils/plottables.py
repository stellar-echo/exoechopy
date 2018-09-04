"""
This module provides base classes for objects that can be plotted with the visualize tools.
"""

from .globals import mpl_colors

__all__ = ['Plottable']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class Plottable:
    """Mixin that describes how to plot an object."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 point_color=None,
                 point_size=None,
                 path_color=None,
                 linewidth=None,
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
