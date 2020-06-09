import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one, time_diff
from ....helpers.rotation import angles2dcm

thrusttable_spec = empty_spec + \
    [('times', nb.float64[:]),

     ('thrusts', nb.float64[:]),

     ('thrust_misalignment', nb.float64[:]),
     ('nozzle_pos', nb.float64[:]),
     ('burn_time', nb.float64),

     ('thrust_scale', nb.float64),
     ('damping', nb.boolean)]


@nb.jitclass(thrusttable_spec)
class ThrustTable(object):
    def __init__(self):
        self.id = 'ThrustTable'
        self.type = 'Thrust'

        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.times = np.zeros(1, dtype=np.float64)
        self.thrusts = np.zeros(1, dtype=np.float64)

        self.thrust_misalignment = np.zeros(3, dtype=np.float64) # rads

        self.nozzle_pos = np.zeros(3, dtype=np.float64)

        self.burn_time = 1e3
        self.thrust_scale = 1
        self.damping = False

    def rhs(self, Core):
        if Core.t > self.burn_time:
            return

        thrust = interp_one(Core.t, self.times, self.thrusts) * self.thrust_scale
        if thrust < 0:
            thrust = 0

        r_e = self.nozzle_pos - Core.cg
        # r_e = self.nozzle_pos - np.array([Core.cg[0], 0., 0.])

        T = np.dot(angles2dcm(np.array([self.thrust_misalignment[0],
                              self.thrust_misalignment[1],
                              self.thrust_misalignment[2]]), unit="deg"),
                   np.array([thrust, 0., 0.]))

        M = np.cross(r_e, T)

        if self.damping:
            Mjetd = -Core.mdot*np.cross(r_e, np.cross(Core.omega, r_e))
            M += Mjetd

        if Core.t > self.burn_time:
            T *= 0
            M *= 0
        
        Core.force += T
        Core.moment += M

        if Core.logging:
            self.history["f_t[0]"] = np.append(self.history["f_t[0]"], T[0])
            self.history["f_t[1]"] = np.append(self.history["f_t[1]"], T[1])
            self.history["f_t[2]"] = np.append(self.history["f_t[2]"], T[2])

            self.history["m_t[0]"] = np.append(self.history["m_t[0]"], M[0])
            self.history["m_t[1]"] = np.append(self.history["m_t[1]"], M[1])
            self.history["m_t[2]"] = np.append(self.history["m_t[2]"], M[2])



    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["f_t[0]"] = np.zeros(1, dtype=np.float64)
        self.history["f_t[1]"] = np.zeros(1, dtype=np.float64)
        self.history["f_t[2]"] = np.zeros(1, dtype=np.float64)

        self.history["m_t[0]"] = np.zeros(1, dtype=np.float64)
        self.history["m_t[1]"] = np.zeros(1, dtype=np.float64)
        self.history["m_t[2]"] = np.zeros(1, dtype=np.float64)




# Vars to take into account :
# - thrust
# - nozzle_diameter
# - P_ref
# - nozzle_pos
# - burn_time
# - thrust_scale
# - thrust_scale_end_time
# - expansion_ratio