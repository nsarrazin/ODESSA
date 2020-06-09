import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type

gravityJ2_spec = empty_spec + [("mu", nb.float64),
                               ("J2", nb.float64),
                               ("re", nb.float64)]


@nb.jitclass(gravityJ2_spec)
class GravityJ2(object):
    def __init__(self):
        self.id = 'GravityJ2'
        self.type = 'Gravity'
        # self.mu =  3.986004418e14
        self.mu = 1.40764431100E+16 * 0.3048**3
        self.J2 = 1.08262982000000E-03
        self.re = 20925646.9951*0.3048
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        x_norm = np.linalg.norm(Core.pos)
        pos_i = Core.DCMci.T @ Core.pos

        factor = 1 - 3*self.J2*self.re**2 * \
            (5*pos_i[2]**2 - x_norm**2)/(2*x_norm**4)
        g = -(self.mu*(pos_i)/x_norm**3)*factor
        f_g = (Core.DCMci @ g)*Core.mass
        Core.force[0] += f_g[0]
        Core.force[1] += f_g[1]
        Core.force[2] += f_g[2]

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

@nb.jitclass(gravityJ2_spec)
class GravityJ26DoF(object):
    def __init__(self):
        self.id = 'GravityJ26DoF'
        self.type = 'Gravity'
        # self.mu =  3.986004418e14
        self.mu = 1.40764431100E+16 * 0.3048**3
        self.J2 = 1.08262982000000E-03
        self.re = 20925646.9951*0.3048
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        x_norm = np.linalg.norm(Core.pos)
        pos_i = Core.DCMci.T @ Core.pos

        factor = 1 - 3*self.J2*self.re**2 * \
            (5*pos_i[2]**2 - x_norm**2)/(2*x_norm**4)
        g = -(self.mu*(pos_i)/x_norm**3)*factor
        f_g = (Core.DCMbc @ Core.DCMci @ g)*Core.mass
        Core.force[0] += f_g[0]
        Core.force[1] += f_g[1]
        Core.force[2] += f_g[2]

        if Core.logging:
            self.history["g[0]"] = np.append(self.history["g[0]"], f_g[0]/Core.mass)
            self.history["g[1]"] = np.append(self.history["g[1]"], f_g[1]/Core.mass)
            self.history["g[2]"] = np.append(self.history["g[2]"], f_g[2]/Core.mass)

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["g[0]"] = np.zeros(1, dtype=np.float64)
        self.history["g[1]"] = np.zeros(1, dtype=np.float64)
        self.history["g[2]"] = np.zeros(1, dtype=np.float64)