import numba as nb
import numpy as np
from ....helpers.frames import ecef_to_wgs84, wgs84_to_ecef
from ....helpers.transformations import ecef2spherical, TCI, TEC

from ...Empty import empty_spec, float_array_type

core3DoF_spec = empty_spec + [("force", nb.float64[:]),
                              ("_pos", nb.float64[:]),
                              ("vel", nb.float64[:]),
                              ("acc", nb.float64[:]),
                              ("mass", nb.float64),
                              ("mdot", nb.float64),
                              ("t", nb.float64),

                              ("rho", nb.float64),
                              ("a", nb.float64),
                              ("p", nb.float64),
                              ("w_e", nb.float64),

                              ("_lla", nb.float64[:]),

                              ("y_old", nb.float64[:]),
                              ("y0", nb.float64[:]),
                              ("logging", nb.boolean),

                              ("DCMci", nb.float64[:, :]),
                              ("DCMec", nb.float64[:, :]),
                              ]


@nb.jitclass(core3DoF_spec)
class Core3DoF(object):
    def __init__(self):
        self.id = 'Core3DoF'
        self.type = 'Core'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.force = np.zeros(3, dtype=np.float64)
        self.mass = 0.
        self.mdot = 0

        self.t = 0.

        self.rho = 0
        self.a = 0
        self.p = 0
        self.w_e = 0.00007292115054

        self._pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.acc = np.zeros(3, dtype=np.float64)

        self._lla = np.zeros(3, dtype=np.float64)
        self.y_old = np.zeros(6, dtype=np.float64)
        self.y0 = np.zeros(6, dtype=np.float64)

        # from Inertial frame of reference to Earth centered frame of reference
        self.DCMci = np.zeros((3, 3), dtype=np.float64)
        # from Earth centered frame of reference to Vehicle carried normal earth frame
        self.DCMec = np.zeros((3, 3), dtype=np.float64)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, var):
        self._pos = var
        self._lla = ecef_to_wgs84(var)
        self.update_DCM()

    @property
    def lla(self):
        return self._lla

    @lla.setter
    def lla(self, var):
        self._lla = var
        self._pos = wgs84_to_ecef(var)
        self.update_DCM()

    def reset(self):
        self.force = np.zeros(3, dtype=np.float64)
        self.mass = 0.
        self.mdot = 0

        self.t = 0.
        self.rho = 0
        self.a = 0
        self.w_e = 0.00007292115054

        self.vel = np.zeros(3, dtype=np.float64)
        self.acc = np.zeros(3, dtype=np.float64)

        self._pos = np.zeros(3, dtype=np.float64)
        self._lla = np.zeros(3, dtype=np.float64)
        self.y_old = np.zeros(6, dtype=np.float64)

    @property
    def y(self):
        return np.hstack((self._pos, self.vel))

    @y.setter
    def y(self, y):
        self.y_old = y
        self.pos = np.array([y[0], y[1], y[2]])
        self.vel = np.array([y[3], y[4], y[5]])

    @property
    def dy(self):
        return np.hstack((self.vel, self.acc))

    def update_DCM(self):
        # Frames of references
        # * **i** Inertial reference frame
        # * **c** Earth centred reference frame
        # * **e** Vehicle carried normal earth frame
        # * **b** Body-fixed reference frame
        # * **a** Aerodynamic (air-path) reference frame

        r, delta, tau = ecef2spherical(self.pos)
        self.DCMci = TCI(self.t, omega_t=self.w_e)
        self.DCMec = TEC(tau, delta)
