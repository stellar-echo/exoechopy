
"""
This module provides interfaces for simplifying interactions with astropy objects.
"""

from astropy import units as u

__all__ = ['u_str', 'u_labelstr']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def u_str(quantity):
    return "{0}".format(quantity)


def u_labelstr(quantity: u.Quantity,
               add_parentheses: bool=False) -> str:
    """Extracts label from a quantity, then converts it to a latex_inline string for plot labels

    Optionally, can add parentheses around the unit (just a shortcut, cuz you're gonna want them anyway)

    Parameters
    ----------
    quantity
        An astropy quantity
    add_parentheses
        Optionally add parentheses around the quantity, e.g., (m/s) instead of m/s

    Returns
    -------
    str
        String derived from the quantity unit
    """
    if isinstance(quantity, u.Quantity):
        if add_parentheses:
            return "("+quantity.unit.to_string('latex_inline')+")"
        else:
            return quantity.unit.to_string('latex_inline')
    else:
        return ""
