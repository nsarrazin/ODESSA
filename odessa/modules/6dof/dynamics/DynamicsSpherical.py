import numpy as np
import numba as nb
from ...Empty import empty_spec, float_array_type
from ....helpers.rotation import dcm2angles
dynamicsspherical_spec = empty_spec + []


@nb.experimental.jitclass(dynamicsspherical_spec)
class DynamicsSpherical(object):
    """The full 6DoF dynamics module.

        Position is in ECEF with forces in body frame.
        Moments and angular rotations also in bodyframe.
    """
    def __init__(self):
        self.id = 'DynamicsSpherical'
        self.type = 'Dynamics'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)
        self.init_history()

    def rhs(self, Core):
        omega = Core.omega
        Vc = Core.vel

        omega_t = np.array([0, 0, Core.w_e])

        M = Core.moment
        # print(Core.Idot)
        omegadot = np.linalg.inv(Core.inertia).dot((M - np.cross(omega, Core.inertia.dot(omega))
                                                    - Core.Idot.dot(omega)))
        # omegadot = np.linalg.inv(Core.inertia).dot((M - np.cross(omega, Core.inertia.dot(omega))
                                                    # ))

        Ac = Core.DCMbc.T @ Core.force/Core.mass - (2*np.cross(omega_t, Vc)) - np.cross(omega_t, np.cross(omega_t, Core.pos))
        # Ac = Core.DCMbc.T @ Core.force/Core.mass - (2*np.cross(omega_t, Vc))
        # Ac = Core.DCMbc.T @ Core.force/Core.mass

        Core.omegadot = omegadot
        Core.acc = Ac

        if Core.logging:
            x_i = Core.DCMci.T @ Core.pos
            x_e = Core.DCMec @ Core.pos
            v_b = Core.DCMbc @ Core.vel
            v_i = Core.DCMci.T @ Core.vel + np.cross(np.array([0., 0., Core.w_e]), Core.pos)
            a_b = Core.DCMbc @ Core.acc
            a_c = Core.acc

            dcm_be = Core.DCMbe
            angles_be = dcm2angles(dcm_be)
            # angles_be = Core.dcm.T @ np.array([-1., 0., 0.])

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

            self.history["v_b[0]"] = np.append(self.history["v_b[0]"], v_b[0])
            self.history["v_b[1]"] = np.append(self.history["v_b[1]"], v_b[1])
            self.history["v_b[2]"] = np.append(self.history["v_b[2]"], v_b[2])

            self.history["v_i[0]"] = np.append(self.history["v_i[0]"], v_i[0])
            self.history["v_i[1]"] = np.append(self.history["v_i[1]"], v_i[1])
            self.history["v_i[2]"] = np.append(self.history["v_i[2]"], v_i[2])

            self.history["a_b[0]"] = np.append(self.history["a_b[0]"], a_b[0])
            self.history["a_b[1]"] = np.append(self.history["a_b[1]"], a_b[1])
            self.history["a_b[2]"] = np.append(self.history["a_b[2]"], a_b[2])

            self.history["a_c[0]"] = np.append(self.history["a_c[0]"], a_c[0])
            self.history["a_c[1]"] = np.append(self.history["a_c[1]"], a_c[1])
            self.history["a_c[2]"] = np.append(self.history["a_c[2]"], a_c[2])

            self.history["f[0]"] = np.append(self.history["f[0]"], Core.force[0])
            self.history["f[1]"] = np.append(self.history["f[1]"], Core.force[1])
            self.history["f[2]"] = np.append(self.history["f[2]"], Core.force[2])

            self.history["m[0]"] = np.append(self.history["m[0]"], Core.moment[0])
            self.history["m[1]"] = np.append(self.history["m[1]"], Core.moment[1])
            self.history["m[2]"] = np.append(self.history["m[2]"], Core.moment[2])

            self.history["omega[0]"] = np.append(self.history["omega[0]"], Core.omega[0])
            self.history["omega[1]"] = np.append(self.history["omega[1]"], Core.omega[1])
            self.history["omega[2]"] = np.append(self.history["omega[2]"], Core.omega[2])

            self.history["dcm_be[0,0]"] = np.append(self.history["dcm_be[0,0]"], dcm_be[0,0])
            self.history["dcm_be[0,1]"] = np.append(self.history["dcm_be[0,1]"], dcm_be[0,1])
            self.history["dcm_be[0,2]"] = np.append(self.history["dcm_be[0,2]"], dcm_be[0,2])
            self.history["dcm_be[1,0]"] = np.append(self.history["dcm_be[1,0]"], dcm_be[1,0])
            self.history["dcm_be[1,1]"] = np.append(self.history["dcm_be[1,1]"], dcm_be[1,1])
            self.history["dcm_be[1,2]"] = np.append(self.history["dcm_be[1,2]"], dcm_be[1,2])
            self.history["dcm_be[2,0]"] = np.append(self.history["dcm_be[2,0]"], dcm_be[2,0])
            self.history["dcm_be[2,1]"] = np.append(self.history["dcm_be[2,1]"], dcm_be[2,1])
            self.history["dcm_be[2,2]"] = np.append(self.history["dcm_be[2,2]"], dcm_be[2,2])

            self.history["psi"] = np.append(self.history["psi"], angles_be[0])
            self.history["theta"] = np.append(self.history["theta"], angles_be[1])
            self.history["phi"] = np.append(self.history["phi"], angles_be[2])

            self.history["tower_exit_velocity"] = np.append(self.history["tower_exit_velocity"], v_b[0])

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)
        # LLA [x,y,z], x_i [x,y,z], v_b [x,y,z], v_i [x,y,z], a_b [x, y, z], omega [p,q,r], q [Pa]

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

        self.history["v_b[0]"] = np.zeros(1, dtype=np.float64)
        self.history["v_b[1]"] = np.zeros(1, dtype=np.float64)
        self.history["v_b[2]"] = np.zeros(1, dtype=np.float64)

        self.history["v_i[0]"] = np.zeros(1, dtype=np.float64)
        self.history["v_i[1]"] = np.zeros(1, dtype=np.float64)
        self.history["v_i[2]"] = np.zeros(1, dtype=np.float64)

        self.history["a_b[0]"] = np.zeros(1, dtype=np.float64)
        self.history["a_b[1]"] = np.zeros(1, dtype=np.float64)
        self.history["a_b[2]"] = np.zeros(1, dtype=np.float64)

        self.history["a_c[0]"] = np.zeros(1, dtype=np.float64)
        self.history["a_c[1]"] = np.zeros(1, dtype=np.float64)
        self.history["a_c[2]"] = np.zeros(1, dtype=np.float64)

        self.history["f[0]"] = np.zeros(1, dtype=np.float64)
        self.history["f[1]"] = np.zeros(1, dtype=np.float64)
        self.history["f[2]"] = np.zeros(1, dtype=np.float64)

        self.history["m[0]"] = np.zeros(1, dtype=np.float64)
        self.history["m[1]"] = np.zeros(1, dtype=np.float64)
        self.history["m[2]"] = np.zeros(1, dtype=np.float64)

        self.history["omega[0]"] = np.zeros(1, dtype=np.float64)
        self.history["omega[1]"] = np.zeros(1, dtype=np.float64)
        self.history["omega[2]"] = np.zeros(1, dtype=np.float64)

        self.history["dcm_be[0,0]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[0,1]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[0,2]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[1,0]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[1,1]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[1,2]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[2,0]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[2,1]"] = np.zeros(1, dtype=np.float64)
        self.history["dcm_be[2,2]"] = np.zeros(1, dtype=np.float64)

        self.history["psi"] = np.zeros(1, dtype=np.float64)
        self.history["theta"] = np.zeros(1, dtype=np.float64)
        self.history["phi"] = np.zeros(1, dtype=np.float64)

        self.history["tower_exit_velocity"] = np.zeros(1, dtype=np.float64)
