import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type

gravityRTS_spec = empty_spec + [("mu", nb.float64)]

@nb.jitclass(gravityRTS_spec)
class Gravity6DoF(object):
    """
    The gravity module in body frame following the inverse square law.
    """
    def __init__(self):
        self.id = 'Gravity6DoF'
        self.type = 'Gravity'
        self.mu = 0.3986004415e15
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        x_norm = np.linalg.norm(Core.pos)
        g = self.mu / np.power(x_norm, 2)
        f_g_e = np.array([0, 0, g*Core.mass])

        f_g = Core.DCMbe.dot(f_g_e)

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
