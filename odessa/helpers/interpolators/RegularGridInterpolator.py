import numba as nb
from math import floor
import numpy as np
from odessa.helpers.math_func import clip, lerp

regular_grid_interp = [
    ("values", nb.float64[:, :]),
    ("d0x0", nb.float64),
    ("d1x0", nb.float64),
    ("d0dx", nb.float64),
    ("d1dx", nb.float64)
]

@nb.jitclass(regular_grid_interp)
class RegularGridInterpolator(object):
    ''' 2D grid interpolator which returns a linearly interpolated value using patch or grid interpolation.
    The interpolator assumes the derivative, or the stepsize over any axis to be constant.


    '''

    def __init__(self):
        self.values = np.ones((3,3), dtype=np.float64)

        # 'd': dimension - first number indicating which dimension, 'x': indicating the coordinate value - second number
        # indicates the nth number along the vector axis., if 'dx', then it indicates a derivative or step size.
        self.d0x0 = 0.
        self.d1x0 = 0.
        self.d0dx = 0.2
        self.d1dx = 1.

    def eval(self, x):
        '''

        :param x: the coordinates of the to-be-interpolated value; must have shape (2,)
        :return:
        '''

        # A     E   B
        #  +----+--+
        #  |    |  |
        #  |   G+  |
        #  |    |  |
        #  |    |  |
        #  +----+--+
        # C     F   D

        d0xi = floor((x[0] - self.d0x0) / self.d0dx)
        d1xi = floor((x[1] - self.d1x0) / self.d1dx)

        # clip the input vector if it is out-of-bounds
        d0xi = clip(d0xi, 0, self.values.shape[0] - 2)
        d1xi = clip(d1xi, 0, self.values.shape[1] - 2)

        # compute the normalized distance between x and AB, and x and AC
        d0xt = (x[0] - (self.d0x0 + d0xi * self.d0dx)) / self.d0dx
        d1xt = (x[1] - (self.d1x0 + d1xi * self.d1dx)) / self.d1dx

        # access and assign the values of the ABCD points.
        value_a = self.values[d0xi, d1xi]
        value_b = self.values[d0xi, d1xi + 1]
        value_c = self.values[d0xi + 1, d1xi]
        value_d = self.values[d0xi + 1, d1xi + 1]

        value_e = lerp(
            t=d1xt,
            y0=value_a,
            y1=value_b)

        value_f = lerp(
            t=d1xt,
            y0=value_c,
            y1=value_d)

        return lerp(
            t=d0xt,
            y0=value_e,
            y1=value_f)
