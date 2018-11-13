
"""
This module provides interfaces for simplifying interactions with astropy objects.
"""

from astropy import units as u

__all__ = ['u_str', 'u_labelstr',
           'to_quantity', 'to_quantity_list']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Presentation and print support
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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Import support
def to_quantity(val_unit_tuple: tuple) -> u.Quantity:
    """Helps import values with specific units, typically from external or config files

    Parameters
    ----------
    val_unit_tuple
        (value, unit) to be converted into an astropy Quantity

    Returns
    -------
    u.Quantity
    """
    try:
        val, unit = val_unit_tuple
    except TypeError:
        raise TypeError("Cannot unpack unit, must be in form (value, unit)")
    return u.Quantity(val, unit)


def to_quantity_list(val_unit_tuple_list):
    return [to_quantity(v_u) for v_u in val_unit_tuple_list]
