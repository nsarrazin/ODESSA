import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type

mass_spec = empty_spec + \
    [("mass", nb.float64)]

massinertia_spec = mass_spec + [("inertia", nb.float64[:, :])]
@nb.experimental.jitclass(massinertia_spec)
class ConstantMassInertia(object):
    def __init__(self):
        self.id = 'ConstantMassInertia'
        self.type = 'Mass'

        self.mass = 0
        self.inertia = np.eye(3, dtype=np.float64)

    def rhs(self, Core):
        Core.mass = self.mass
        Core.inertia = self.inertia

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)
