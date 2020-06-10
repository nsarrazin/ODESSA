import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type

mass_spec = empty_spec + \
    [("mass", nb.float64)]


@nb.jitclass(mass_spec)
class ConstantMass(object):
    def __init__(self):
        self.id = 'ConstantMass'
        self.type = 'Mass'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.mass = 0

    def rhs(self, Core):
        Core.mass = self.mass

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)