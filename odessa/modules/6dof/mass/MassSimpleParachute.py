import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one, time_diff

mass_spec = empty_spec + \
    [("mass", nb.float64[:])]

simple_parachute_mass_spec = mass_spec + [
    ('Ixx', nb.float64[:]),
    ('Iyy', nb.float64[:]),
    ('Izz', nb.float64[:]),
    ('Ixy', nb.float64[:]),
    ('Ixz', nb.float64[:]),
    ('Iyz', nb.float64[:]),

    ('cg_x', nb.float64[:]),
    ('cg_y', nb.float64[:]),
    ('cg_z', nb.float64[:]),

    ('empty_mass_scale', nb.float64),
    ("extra_mass", nb.float64)
]

@nb.jitclass(simple_parachute_mass_spec)
class MassSimpleParachute(object):
    def __init__(self):
        self.id = 'MassSimpleParachute'
        self.type = 'Mass'

        self.mass = np.ones(1, dtype=np.float64)

        self.Ixx = np.zeros(1, dtype=np.float64)
        self.Iyy = np.zeros(1, dtype=np.float64)
        self.Izz = np.zeros(1, dtype=np.float64)
        self.Ixy = np.zeros(1, dtype=np.float64)
        self.Ixz = np.zeros(1, dtype=np.float64)
        self.Iyz = np.zeros(1, dtype=np.float64)

        self.cg_x = np.ones(1, dtype=np.float64)
        self.cg_y = np.ones(1, dtype=np.float64)
        self.cg_z = np.ones(1, dtype=np.float64)

        self.empty_mass_scale = 1.
        self.extra_mass = 0.
    def rhs(self, Core):

        Ixx = self.Ixx[0]
        Iyy = self.Iyy[0]
        Izz = self.Izz[0]
        Ixy = self.Ixy[0]
        Ixz = self.Ixz[0]
        Iyz = self.Iyz[0]


        inertia = np.array([[Ixx, Ixy, Ixz],
                            [Ixy, Iyy, Iyz],
                            [Ixz, Iyz, Izz]])

        Idot = np.zeros((3, 3), dtype=np.float64)
        cg = np.array([self.cg_x[0], self.cg_y[0], self.cg_z[0]])

        Core.mass = self.mass[0] * self.empty_mass_scale + self.extra_mass
        Core.inertia = inertia
        Core.cg = cg
        Core.Idot = Idot

        if Core.logging:
            self.history["cg_v[0]"] = np.append(self.history["cg_v[0]"], self.cg_x[0])
            self.history["cg_v[1]"] = np.append(self.history["cg_v[1]"], self.cg_y[1])
            self.history["cg_v[2]"] = np.append(self.history["cg_v[2]"], self.cg_z[2])

            self.history["mass_v"] = np.append(self.history["mass_v"], Core.mass)

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["cg_v[0]"] = np.zeros(1, dtype=np.float64)
        self.history["cg_v[1]"] = np.zeros(1, dtype=np.float64)
        self.history["cg_v[2]"] = np.zeros(1, dtype=np.float64)

        self.history["mass_v"] = np.zeros(1, dtype=np.float64)