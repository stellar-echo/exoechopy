
"""
Supports interpretation of probability distribution functions (PDF) during imports
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from scipy import stats
from .constants import *
from scipy.stats._continuous_distns import _distn_names as distribution_names

__all__ = ["parse_pdf", "parse_pdf_kwargs", "pdf_dict"]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def parse_pdf(pdf_candidate, warnings_on=True):
    pdf_unit = None
    if isinstance(pdf_candidate, CountType):
        pdf = pdf_candidate
        if isinstance(pdf_candidate, u.Quantity):
            # Strip unit, recombine later (just to be consistent between types)
            pdf = pdf_candidate.value
            pdf_unit = pdf_candidate.unit
    elif isinstance(pdf_candidate, (list, tuple)):
        if len(pdf_candidate) > 1:
            if isinstance(pdf_candidate[1], CountType):
                pdf = stats.uniform(loc=pdf_candidate[0], scale=pdf_candidate[1] - pdf_candidate[0])
            elif isinstance(pdf_candidate[1], u.UnitBase):
                # Strip list structure, call recursively:
                pdf, _ = parse_pdf(pdf_candidate[0])
                pdf_unit = pdf_candidate[1]
            else:
                if warnings_on:
                    warnings.warn("pdf_candidate is list with unanticipated type, proceed with caution", AstropyUserWarning)
                pdf = pdf_candidate
        else:
            # Strip list structure, call recursively:
            pdf, _ = parse_pdf(pdf_candidate[0])
    elif isinstance(pdf_candidate, RVFrozen):
        pdf = pdf_candidate
    else:
        if warnings_on:
            warnings.warn("pdf_candidate is unanticipated type, proceed with caution", AstropyUserWarning)
        pdf = pdf_candidate
    return pdf, pdf_unit


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def parse_pdf_kwargs(_kwargs):
    """Interprets custom kwargs to pass flexible options to underlying flares

    Parameters
    ----------
    _kwargs

    Returns
    -------
    (Arguments to be passed, argument type (e.g., float), the keywords, astropy units associated with arg)
    """
    arg_type_list = []
    new_args = []
    new_kw = []
    kw_units = []
    # Keep units in registration with different kwarg types such as floats, ints, ranges, and distribution funcs
    for (kw, val) in _kwargs.items():
        if isinstance(val, CountType):
            if isinstance(val, u.Quantity):
                if isinstance(val.value, np.ndarray):
                    new_args.append(stats.uniform(loc=val[0], scale=val[1] - val[0]))
                else:
                    new_args.append(val.value)
                kw_units.append(val.unit)
            else:
                new_args.append(val)
                kw_units.append(1.)
            arg_type_list.append((str(kw), 'float64'))
        elif isinstance(val, (list, tuple)):
            if isinstance(val[1], (str, u.IrreducibleUnit)):
                kw_units.append(val[1])
                if isinstance(val[0], (list, tuple)):
                    new_args.append(stats.uniform(loc=val[0][0], scale=val[0][1] - val[0][0]))
                else:
                    new_args.append(val[0])
            else:
                new_args.append(stats.uniform(loc=val[0], scale=val[1] - val[0]))
                kw_units.append(1.)
            arg_type_list.append((str(kw), 'float64'))
        elif isinstance(val, RVFrozen):
            arg_type_list.append((str(kw), 'float64'))
            new_args.append(val)
            kw_units.append(1.)
        else:
            if isinstance(val, str):
                string_length = 16
                # For use in a structured numpy array:
                arg_type_list.append((str(kw), 'U' + str(string_length)))
                if len(val) > string_length:
                    AstropyUserWarning("Structured numpy array only configured to accept U" + str(
                        string_length) + " strings, received "
                                       + val + ", converting to " + val[:string_length])
                    val = val[:string_length]
                new_args.append(val)
                kw_units.append("str")
            else:
                arg_type_list.append(type(val))
                new_args.append(val)
                kw_units.append(1)

        if kw[-4:] == '_pdf':
            new_kw.append(kw[:-4])
        else:
            new_kw.append(kw)

    return new_args, arg_type_list, new_kw, kw_units

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


pdf_dict = {dist_name: getattr(stats, dist_name) for dist_name in distribution_names}
