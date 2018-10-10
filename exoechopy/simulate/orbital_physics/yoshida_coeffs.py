
"""
This module provides 6th order Yoshida symplectic integration coefficients (Solution A).

References
----------
Yoshida, H. (1990). "Construction of higher order symplectic integrators". Phys. Lett. A. 150 (5–7): 262.
https://en.wikipedia.org/wiki/Symplectic_integrator

"""

import numpy as np

__all__ = ['c_vect', 'd_vect']

# --------------------- #
w1 = -.117767998417887E1
w2 = 0.235573213359357E0
w3 = 0.784513610477560E0
w0 = 1-2*(w1 + w2 + w3)

# --------------------- #
k = 8
m = 3

d1 = w3
d7 = d1

d2 = w2
d6 = d2

d3 = w1
d5 = d3

d4 = w0

d8 = 0.

# --------------------- #
c1 = .5*w3
c8 = c1

c2 = .5*(w3 + w2)
c7 = c2

# Verify:
c3 = .5*(w2 + w1)
c6 = c3

c4 = .5*(w1 + w0)
c5 = c4

# --------------------- #

c_vect = np.array([c1, c2, c3, c4, c5, c6, c7, c8])
d_vect = np.array([d1, d2, d3, d4, d5, d6, d7, d8])
