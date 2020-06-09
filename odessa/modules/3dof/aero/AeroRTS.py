import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one

aeroRTS_spec = empty_spec + \
    [("wind_alts", nb.float64[:]),
     ("wind_speeds", nb.float64[:, :]),

     ("drags", nb.float64[:]),
     ("machs", nb.float64[:]), ]


@nb.jitclass(aeroRTS_spec)
class AeroRTS(object):
    def __init__(self):
        self.id = 'AeroRTS'
        self.type = 'Aero'

        self.drags = np.zeros(1, dtype=np.float64)
        self.machs = np.zeros(1, dtype=np.float64)

        self.wind_alts = np.zeros(1, dtype=np.float64)
        self.wind_speeds = np.eye(2, dtype=np.float64)  # radians

    def rhs(self, Core):
        h_geopot = 6371000.0 * Core.lla[2] / (6371000.0 + Core.lla[2])

        _vn, _ve = interp_one(h_geopot, self.wind_alts, self.wind_speeds)

        wind_local = np.array([_vn,
                               _ve,
                               0])


        wind_c = Core.DCMec.T @ wind_local
        v_tas = Core.vel + wind_c

        v = np.sqrt(v_tas[0]**2 + v_tas[1]**2 + v_tas[2]**2)
        dyn_press = 0.5 * Core.rho * v ** 2
        if v == 0.:
            return
        mach = v/Core.a

        drag = interp_one(mach, self.machs, self.drags)

        F_drag = -1*dyn_press*drag

        aero_force = F_drag*v_tas/v

        Core.force[0] += aero_force[0]
        Core.force[1] += aero_force[1]
        Core.force[2] += aero_force[2]

        if Core.logging:
            self.history["f_a[0]"] = np.append(self.history["f_a[0]"], aero_force[0])
            self.history["f_a[1]"] = np.append(self.history["f_a[1]"], aero_force[1])
            self.history["f_a[2]"] = np.append(self.history["f_a[2]"], aero_force[2])
            self.history["M"] = np.append(self.history["M"], mach)
            self.history["q"] = np.append(self.history["q"], dyn_press)

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["M"] = np.zeros(1, dtype=np.float64)
        self.history["q"] = np.zeros(1, dtype=np.float64)

        self.history["f_a[0]"] = np.zeros(1, dtype=np.float64)
        self.history["f_a[1]"] = np.zeros(1, dtype=np.float64)
        self.history["f_a[2]"] = np.zeros(1, dtype=np.float64)