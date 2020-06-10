import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one


constant_thrust_spec = empty_spec + \
    [('thrust', nb.float64)]


@nb.experimental.jitclass(constant_thrust_spec)
class ConstantThrust(object):
    def __init__(self):
        self.id = 'ConstantThrust'
        self.type = 'Thrust'
        self.thrust = 0.
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        f_thrust_bf = np.array([self.thrust, 0, 0])
        Core.force[0] += f_thrust_bf[0]
        Core.force[1] += f_thrust_bf[1]
        Core.force[2] += f_thrust_bf[2]


interpolation_thrust_spec = empty_spec + \
    [('thrusts', nb.float64[:]),
     ['times', nb.float64[:]],
     ['x_thrust', nb.float64[:]]]


@nb.experimental.jitclass(interpolation_thrust_spec)
class InterpolationThrust(object):
    def __init__(self):
        self.id = 'InterpolationThrust'
        self.type = 'Thrust'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.thrusts = np.zeros(1, dtype=np.float64)

        self.times = np.zeros(1, dtype=np.float64)

        self.x_thrust = np.zeros(3, dtype=np.float64)

    def rhs(self, Core):
        thrust = interp_one(Core.t, self.times, self.thrusts)
        f_thrust_bf = np.array([thrust, 0, 0])

        Core.force[0] += f_thrust_bf[0]
        Core.force[1] += f_thrust_bf[1]
        Core.force[2] += f_thrust_bf[2]

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)
