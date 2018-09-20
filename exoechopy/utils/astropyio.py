
"""
This module provides interfaces for simplifying interactions with astropy objects.
"""

from astropy import units as u

__all__ = ['u_str', 'u_labelstr']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def u_str(quantity):
    return "{0}".format(quantity)


def u_labelstr(quantity):
    if isinstance(quantity, u.Quantity):
        return quantity.unit.to_string('latex_inline')
    else:
        return ""
