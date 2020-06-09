import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.transformations import TbE, TEC, TCI

tower_spherical_spec = empty_spec + \
    [('heading', nb.float64),
     ('elevation', nb.float64),
     ('tower_length', nb.float64),
     ('roll', nb.float64)]


@nb.jitclass(tower_spherical_spec)
class TowerSpherical(object):
    def __init__(self):
        self.id = 'TowerSpherical'
        self.type = 'Tower'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.heading = 0.
        self.elevation = 0.
        self.tower_length = 0.
        self.roll = 0.

    def rhs(self, Core):
        #XXX: This needs a rewrite for sure
        if self.tower_length > 0.:
            DCMbe = TbE(np.radians(self.heading), np.radians(
                self.elevation), np.radians(self.roll))

            DCMbc = DCMbe @ Core.DCMec

            omega_t = np.array([0., 0., Core.w_e])
            # Core.omega = Core.dcm.dot(omega_t)

            Acor_ic_b = DCMbc.dot(2*np.cross(Core.vel, omega_t))
            Acc_b = Core.force/Core.mass + Acor_ic_b
                                                           # + Acor_ic_b
            # print(Acc_b)
            # if Core.lla[2] <= 0:
                # make sure rocket doesn't fall through the earth on the tower
            Acc_b[0] = max(Acc_b[0], 0)

            Acc_b[1] = 0.
            Acc_b[2] = 0.

            Core.force = Core.mass*(Acc_b-Acor_ic_b)
            Core.moment = (np.cross(Core.omega, Core.inertia.dot(Core.omega))) + Core.Idot.dot(Core.omega)
            # print(DCMbc @ Core.DCMci, omega_t)

        else:
            pass

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def on_start(self, Core):
        DCMbe = TbE(np.radians(self.heading), np.radians(self.elevation), np.radians(self.roll))

        Core.DCMec = TEC(Core.lla[1], Core.lla[0])
        Core.DCMci = TCI(Core.t)
        Core.dcm = DCMbe @ Core.DCMec @ Core.DCMci  # DCMbi

        omega_t = np.array([0., 0., Core.w_e]).reshape((3, 1))
        Core.omega = Core.dcm.dot(omega_t)

