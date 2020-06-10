import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type

gravity_spec = empty_spec + [("g", nb.float64)]


@nb.experimental.jitclass(gravity_spec)
class Gravity(object):
    def __init__(self):
        self.id = 'ConstantGravityFlat'
        self.type = 'Gravity'
        self.g = 0.

    def rhs(self, Core):
        Core.force[2] += -1*self.g*Core.mass


gravityRTS_spec = empty_spec + [("mu", nb.float64)]


@nb.experimental.jitclass(gravityRTS_spec)
class GravityRTS(object):
    def __init__(self):
        self.id = 'GravityRTS'
        self.type = 'Gravity'
        self.mu = 3.986004418e14
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        x_norm = np.linalg.norm(Core.pos)
        g = self.mu / np.power(x_norm, 2)
        f_g = -(Core.pos/x_norm)*g*Core.mass

        Core.force[0] += f_g[0]
        Core.force[1] += f_g[1]
        Core.force[2] += f_g[2]

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)