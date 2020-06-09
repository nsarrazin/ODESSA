import numba as nb
import numpy as np
from ....helpers.frames import ecef_to_wgs84, wgs84_to_ecef
from ....helpers.rotation import dcm2angles, dcm2quats, angles2dcm,\
    angles2quats, quats2angles, quats2dcm, \
    omega2qdot, qdot2omega

from ....helpers.transformations import ecef2spherical, TCI, TEC

from ...Empty import float_array_type, empty_spec

core6DoF_spec = empty_spec + [("force", nb.float64[:]),
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
                              ("inertia", nb.float64[:, :]),
                              ("Idot", nb.float64[:, :]),
                              ("_dcm", nb.float64[:, :]),
                              ("_angles", nb.float64[:]),
                              ("_quats", nb.float64[:]),
                              ("_qdot", nb.float64[:]),
                              ("_omega", nb.float64[:]),
                              ("_omegadot", nb.float64[:]),
                              ("moment", nb.float64[:]),
                              ("DCMci", nb.float64[:, :]),
                              ("DCMec", nb.float64[:, :]),
                              ("DCMbe", nb.float64[:, :]),
                              ("DCMbc", nb.float64[:, :]),
                              ("cg", nb.float64[:])
                              ]


@nb.jitclass(core6DoF_spec)
class Core6DoF(object):
    def __init__(self):
        self.id = 'Core6DoF'
        self.type = 'Core'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.t = 0.

        self.mass = 0.
        self.inertia = np.zeros((3, 3), dtype=np.float64)

        self.mdot = 0
        self.Idot = np.zeros((3, 3), dtype=np.float64)

        self.rho = 0
        self.a = 0

        self.w_e = 7.2921235169904e-5

        self._pos = np.zeros(3, dtype=np.float64)
        self._lla = np.zeros(3, dtype=np.float64)

        self.vel = np.zeros(3, dtype=np.float64)
        self.acc = np.zeros(3, dtype=np.float64)

        # rotation
        self._dcm = np.zeros((3, 3), dtype=np.float64)
        self._angles = np.zeros(3, dtype=np.float64)
        self._quats = np.array([1., 0., 0., 0.])
        # angular rates
        self._qdot = np.zeros(4, dtype=np.float64)
        self._omega = np.zeros(3, dtype=np.float64)
        # angular acceleration
        self._omegadot = np.zeros(3, dtype=np.float64)

        self.force = np.zeros(3, dtype=np.float64)
        self.moment = np.zeros(3, dtype=np.float64)

        self.y_old = np.zeros(13, dtype=np.float64)
        # from Vehicle carried normal earth frame to Body-fixed reference frame
        self.DCMbe = np.zeros((3, 3), dtype=np.float64)
        # from Earth centered frame of reference to Body-fixed reference frame
        self.DCMbc = np.zeros((3, 3), dtype=np.float64)
        # self.DCMbi = self.dcm

        self.cg = np.zeros(3, dtype=np.float64)
        self.y0 = np.zeros(13, dtype=np.float64)
        self.logging = False

    # POSITION
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
    # ORIENTATION
    @property
    def dcm(self):
        return self._dcm

    @dcm.setter
    def dcm(self, val):
        self._dcm = val
        self._angles = dcm2angles(val)
        self._quats = dcm2quats(val)

        self.update_DCM()

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, val):
        self._angles = val
        self._dcm = angles2dcm(val)
        self._quats = angles2quats(val)

        self.update_DCM()

    @property
    def quats(self):
        return self._quats

    @quats.setter
    def quats(self, val):
        val = val/np.linalg.norm(val)
        self._angles = quats2angles(val)
        self._dcm = quats2dcm(val)
        self._quats = val

        self.update_DCM()

    # ANGULAR RATES
    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, val):
        self._omega = val
        self._qdot = omega2qdot(val, self._quats)

    @property
    def qdot(self):
        return self._qdot

    @qdot.setter
    def qdot(self, val):
        self._qdot = val
        self._omega = qdot2omega(val, self._quats)

    # ANGULAR ACCELERATION
    @property
    def omegadot(self):
        return self._omegadot

    @omegadot.setter
    def omegadot(self, val):
        self._omegadot = val

    def reset(self):
        self.t = 0.

        self.mass = 0.
        self.inertia = np.zeros((3, 3), dtype=np.float64)
        self.mdot = 0
        self.Idot = np.zeros((3, 3), dtype=np.float64)

        self.rho = 0
        self.a = 0

        self.w_e = 7.2921235169904e-5

        self._pos = np.zeros(3, dtype=np.float64)
        self._lla = np.zeros(3, dtype=np.float64)

        self.vel = np.zeros(3, dtype=np.float64)  # core.vel = ECEF
        self.acc = np.zeros(3, dtype=np.float64)  # core.acc = ECEF

        # rotation
        self._dcm = np.zeros((3, 3), dtype=np.float64)
        self._angles = np.zeros(3, dtype=np.float64)
        self._quats = np.array([1., 0., 0., 0.])
        # angular rates
        self._qdot = np.zeros(4, dtype=np.float64)
        self._omega = np.zeros(3, dtype=np.float64)
        # angular acceleration
        self._omegadot = np.zeros(3, dtype=np.float64)

        self.force = np.zeros(3, dtype=np.float64)  # core.force = bodyframe
        self.moment = np.zeros(3, dtype=np.float64)  # core.moment = bodyframe
        self.y_old = np.zeros(13, dtype=np.float64)

        self.cg = np.zeros(3, dtype=np.float64)

    @property
    def y(self):
        return np.hstack((self.pos, self.vel, self.quats, self.omega))

    @y.setter
    def y(self, y):
        self.y_old = y
        self.pos = np.array([y[0], y[1], y[2]])  # ECEF
        self.vel = np.array([y[3], y[4], y[5]])  # ECEF

        self.quats = np.copy(np.array([y[6], y[7], y[8], y[9]]))
        self.omega = np.copy(np.array([y[10], y[11], y[12]]))


    @property
    def dy(self):
        return np.hstack((self.vel, self.acc, self.qdot, self.omegadot))

    def update_DCM(self):
        # Frames of references
        # * **i** Inertial reference frame
        # * **c** Earth centred reference frame
        # * **e** Vehicle carried normal earth frame
        # * **b** Body-fixed reference frame
        # * **a** Aerodynamic (air-path) reference frame
        # r, delta, tau = ecef2spherical(self.pos)
        delta, tau, r = ecef_to_wgs84(self.pos)
        self.DCMci = TCI(self.t, omega_t=self.w_e)
        self.DCMec = TEC(tau, delta)

        # bi @ ic @ ce = be
        self.DCMbe = self.dcm @ self.DCMci.T @ self.DCMec.T

        self.DCMbc = self.DCMbe @ self.DCMec