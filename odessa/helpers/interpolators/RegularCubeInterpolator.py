import numba as nb
from math import floor
import numpy as np
from odessa.helpers.math_func import clip, lerp

regular_cube_interp = [
    ("values", nb.float64[:, :, :]),
    ("d0x0", nb.float64),
    ("d1x0", nb.float64),
    ("d2x0", nb.float64),
    ("d0dx", nb.float64),
    ("d1dx", nb.float64),
    ("d2dx", nb.float64)
]


@nb.jitclass(regular_cube_interp)
class RegularCubeInterpolator(object):

    def __init__(self):

        self.values = np.ones((1,1,1), dtype=np.float64)
        self.d0x0 = 0.
        self.d1x0 = 0.
        self.d2x0 = 0.
        self.d0dx = 0.
        self.d1dx = 0.
        self.d2dx = 0.


    def eval(self, x):
        '''

        :param x: the coordinates of the to-be-interpolated value; must have shape (3,)
        :return:
        '''

        #  A    E  B                                       A'   E' B'
        #  +----+--+                                       +----+--+
        #  |    |  |                                       |    |  |
        #  |  G +  |                                       | G' +  |
        #  |    |  |                                       |    |  |
        #  |    |  |                                       |    |  |
        #  +----+--+                                       +----+--+
        #  C    F  D                                       C'   F' D'


