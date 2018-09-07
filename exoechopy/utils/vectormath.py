
"""
This module provides functions that are useful in computing values with vectors.
"""

import numpy as np

__all__ = ['angle_between_vectors']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def angle_between_vectors(v1, v2):
    return 2*np.arctan2(np.linalg.norm(np.linalg.norm(v2)*v1-np.linalg.norm(v1)*v2),
                        np.linalg.norm(np.linalg.norm(v2)*v1+np.linalg.norm(v1)*v2))

