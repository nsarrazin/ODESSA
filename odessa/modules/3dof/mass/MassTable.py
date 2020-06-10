import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one, time_diff


masstable6dof_spec = empty_spec + \
    [('times', nb.float64[:]),

     ('masses', nb.float64[:]),

     ('Ixx', nb.float64[:]),
     ('Iyy', nb.float64[:]),
     ('Izz', nb.float64[:]),
     ('Ixy', nb.float64[:]),
     ('Ixz', nb.float64[:]),
     ('Iyz', nb.float64[:]),

     ('cg_x', nb.float64[:]),
     ('cg_y', nb.float64[:]),
     ('cg_z', nb.float64[:]),

     ('cg_shift', nb.float64[:]),
     ('mass_scale', nb.float64),
     ('empty_mass_scale', nb.float64)
     ]


# TODO: Add Mass scale term
# TODO: Verify Idot works
@nb.experimental.jitclass(masstable6dof_spec)
class MassTable6DoF(object):
    def __init__(self):
        self.id = 'MassTable6DoF'
        self.type = 'Mass'

        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.times = np.zeros(1, dtype=np.float64)
        self.masses = np.zeros(1, dtype=np.float64)

        self.Ixx = np.zeros(1, dtype=np.float64)
        self.Iyy = np.zeros(1, dtype=np.float64)
        self.Izz = np.zeros(1, dtype=np.float64)
        self.Ixy = np.zeros(1, dtype=np.float64)
        self.Ixz = np.zeros(1, dtype=np.float64)
        self.Iyz = np.zeros(1, dtype=np.float64)

        self.cg_x = np.zeros(1, dtype=np.float64)
        self.cg_y = np.zeros(1, dtype=np.float64)
        self.cg_z = np.zeros(1, dtype=np.float64)

        self.mass_scale = 1
        self.empty_mass_scale =1
        self.cg_shift = np.zeros(3, dtype=np.float64)

    def rhs(self, Core):
        empty_mass = self.masses[-1]
        mass = interp_one(Core.t, self.times, self.masses)*self.mass_scale + empty_mass * (self.empty_mass_scale - 1)

        mdot = np.abs(time_diff(Core.t, self.times, self.masses*self.mass_scale))

        Ixx = interp_one(Core.t, self.times, self.Ixx)
        Iyy = interp_one(Core.t, self.times, self.Iyy)
        Izz = interp_one(Core.t, self.times, self.Izz)
        Ixy = interp_one(Core.t, self.times, self.Ixy)
        Ixz = interp_one(Core.t, self.times, self.Ixz)
        Iyz = interp_one(Core.t, self.times, self.Iyz)

        Ixx_dot = time_diff(Core.t, self.times, self.Ixx)
        Iyy_dot = time_diff(Core.t, self.times, self.Iyy)
        Izz_dot = time_diff(Core.t, self.times, self.Izz)
        Ixy_dot = time_diff(Core.t, self.times, self.Ixy)
        Ixz_dot = time_diff(Core.t, self.times, self.Ixz)
        Iyz_dot = time_diff(Core.t, self.times, self.Iyz)

        cg_x = interp_one(Core.t, self.times, self.cg_x) + self.cg_shift[0]
        cg_y = interp_one(Core.t, self.times, self.cg_y) + self.cg_shift[1]
        cg_z = interp_one(Core.t, self.times, self.cg_z) + self.cg_shift[2]

        inertia = np.array([[Ixx, Ixy, Ixz],
                            [Ixy, Iyy, Iyz],
                            [Ixz, Iyz, Izz]])

        Idot = np.array([[Ixx_dot, Ixy_dot, Ixz_dot],
                         [Ixy_dot, Iyy_dot, Iyz_dot],
                         [Ixz_dot, Iyz_dot, Izz_dot]])

        cg = np.array([cg_x, cg_y, cg_z])

        Core.mass = mass
        Core.mdot = mdot
        Core.inertia = inertia
        Core.Idot = Idot
        Core.cg = cg

        if Core.logging:
            self.history["cg_v[0]"] = np.append(self.history["cg_v[0]"], cg_x)
            self.history["cg_v[1]"] = np.append(self.history["cg_v[1]"], cg_y)
            self.history["cg_v[2]"] = np.append(self.history["cg_v[2]"], cg_z)

            self.history["mass_v"] = np.append(self.history["mass_v"], Core.mass)

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["cg_v[0]"] = np.zeros(1, dtype=np.float64)
        self.history["cg_v[1]"] = np.zeros(1, dtype=np.float64)
        self.history["cg_v[2]"] = np.zeros(1, dtype=np.float64)

        self.history["mass_v"] = np.zeros(1, dtype=np.float64)
