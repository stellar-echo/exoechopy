"""
This module provides interfaces for simplifying interactions with astropy objects.
"""

from astropy import units as u

__all__ = ['u_str']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def u_str(quantity):
    return "{0}".format(quantity)
