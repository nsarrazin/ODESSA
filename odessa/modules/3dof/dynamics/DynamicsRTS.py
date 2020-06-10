import numpy as np
import numba as nb
from ...Empty import empty_spec, float_array_type
from ....helpers.rotation import dcm2angles
dynamicsRTS_spec = empty_spec + []


@nb.experimental.jitclass(dynamicsRTS_spec)
class DynamicsRTS(object):
    def __init__(self):
        self.id = 'DynamicsRTS'
        self.type = 'Dynamics'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        Vc = Core.vel

        omega_t = np.array([0, 0, Core.w_e])

        Ac = Core.force/Core.mass - (np.cross(2*omega_t, Vc)) - np.cross(omega_t, np.cross(omega_t, Core.pos))

        Core.acc = Ac

        if Core.logging:
            x_i = Core.DCMci.T @ Core.pos
            x_e = Core.DCMec @ Core.pos
            v_i = Core.DCMci.T @ Core.vel + np.cross(np.array([0., 0., Core.w_e]), Core.pos)
            a_c = Core.acc

            self.history["t"] = np.append(self.history["t"], Core.t)

            self.history["x_i[0]"] = np.append(self.history["x_i[0]"], x_i[0])
            self.history["x_i[1]"] = np.append(self.history["x_i[1]"], x_i[1])
            self.history["x_i[2]"] = np.append(self.history["x_i[2]"], x_i[2])

            self.history["x_e[0]"] = np.append(self.history["x_e[0]"], x_e[0])
            self.history["x_e[1]"] = np.append(self.history["x_e[1]"], x_e[1])
            self.history["x_e[2]"] = np.append(self.history["x_e[2]"], x_e[2])

            self.history["lla[0]"] = np.append(self.history["lla[0]"], Core.lla[0])
            self.history["lla[1]"] = np.append(self.history["lla[1]"], Core.lla[1])
            self.history["lla[2]"] = np.append(self.history["lla[2]"], Core.lla[2])

            self.history["v_i[0]"] = np.append(self.history["v_i[0]"], v_i[0])
            self.history["v_i[1]"] = np.append(self.history["v_i[1]"], v_i[1])
            self.history["v_i[2]"] = np.append(self.history["v_i[2]"], v_i[2])

            self.history["a_c[0]"] = np.append(self.history["a_c[0]"], a_c[0])
            self.history["a_c[1]"] = np.append(self.history["a_c[1]"], a_c[1])
            self.history["a_c[2]"] = np.append(self.history["a_c[2]"], a_c[2])

            self.history["f[0]"] = np.append(self.history["f[0]"], Core.force[0])
            self.history["f[1]"] = np.append(self.history["f[1]"], Core.force[1])
            self.history["f[2]"] = np.append(self.history["f[2]"], Core.force[2])

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["t"] = np.zeros(1, dtype=np.float64)

        self.history["x_i[0]"] = np.zeros(1, dtype=np.float64)
        self.history["x_i[1]"] = np.zeros(1, dtype=np.float64)
        self.history["x_i[2]"] = np.zeros(1, dtype=np.float64)

        self.history["x_e[0]"] = np.zeros(1, dtype=np.float64)
        self.history["x_e[1]"] = np.zeros(1, dtype=np.float64)
        self.history["x_e[2]"] = np.zeros(1, dtype=np.float64)

        self.history["lla[0]"] = np.zeros(1, dtype=np.float64)
        self.history["lla[1]"] = np.zeros(1, dtype=np.float64)
        self.history["lla[2]"] = np.zeros(1, dtype=np.float64)

        self.history["v_i[0]"] = np.zeros(1, dtype=np.float64)
        self.history["v_i[1]"] = np.zeros(1, dtype=np.float64)
        self.history["v_i[2]"] = np.zeros(1, dtype=np.float64)

        self.history["a_c[0]"] = np.zeros(1, dtype=np.float64)
        self.history["a_c[1]"] = np.zeros(1, dtype=np.float64)
        self.history["a_c[2]"] = np.zeros(1, dtype=np.float64)

        self.history["f[0]"] = np.zeros(1, dtype=np.float64)
        self.history["f[1]"] = np.zeros(1, dtype=np.float64)
        self.history["f[2]"] = np.zeros(1, dtype=np.float64)