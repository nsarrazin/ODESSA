import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one

aerolinear_spec = empty_spec + \
    [("c_x_0", nb.float64[:]),
     ("c_x_a", nb.float64[:]),
     ("c_x_a2", nb.float64[:]),
     ("c_x_p", nb.float64[:]),
     ("c_x_q", nb.float64[:]),
     ("c_x_r", nb.float64[:]),
     ("c_y_b", nb.float64[:]),
     ("c_y_p", nb.float64[:]),
     ("c_y_q", nb.float64[:]),
     ("c_y_r", nb.float64[:]),
     ("c_z_a", nb.float64[:]),
     ("c_z_p", nb.float64[:]),
     ("c_z_q", nb.float64[:]),
     ("c_z_r", nb.float64[:]),
     ("c_ll_0", nb.float64[:]),
     ("c_ll_p", nb.float64[:]),
     ("c_m_a", nb.float64[:]),
     ("c_m_p", nb.float64[:]),
     ("c_m_q", nb.float64[:]),
     ("c_m_r", nb.float64[:]),
     ("c_ln_b", nb.float64[:]),
     ("c_ln_p", nb.float64[:]),
     ("c_ln_q", nb.float64[:]),
     ("c_ln_r", nb.float64[:]),

     ("machs", nb.float64[:]),
     ("scale", nb.float64[:]),

     ("wind_alt", nb.float64[:]),
     ("wind_speed", nb.float64[:, :]),

     ("x_cp", nb.float64[:]),
     ("lref", nb.float64),
     ("sref", nb.float64)
     ]

MAX_ALPHA = np.radians(10)


@nb.jitclass(aerolinear_spec)
class AeroLinear(object):
    """The 6DoF Linear aerodynamics module.

        Interpolates the aerodynamic coefficients 
        as a function of mach number from a CSV.
    """
    def __init__(self):
        self.id = 'AeroLinear'
        self.type = 'Aero'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.c_x_0 = np.zeros(1, dtype=np.float64)
        self.c_x_a = np.zeros(1, dtype=np.float64)
        self.c_x_a2 = np.zeros(1, dtype=np.float64)
        self.c_x_p = np.zeros(1, dtype=np.float64)
        self.c_x_q = np.zeros(1, dtype=np.float64)
        self.c_x_r = np.zeros(1, dtype=np.float64)
        self.c_y_b = np.zeros(1, dtype=np.float64)
        self.c_y_p = np.zeros(1, dtype=np.float64)
        self.c_y_q = np.zeros(1, dtype=np.float64)
        self.c_y_r = np.zeros(1, dtype=np.float64)
        self.c_z_a = np.zeros(1, dtype=np.float64)
        self.c_z_p = np.zeros(1, dtype=np.float64)
        self.c_z_q = np.zeros(1, dtype=np.float64)
        self.c_z_r = np.zeros(1, dtype=np.float64)
        self.c_ll_0 = np.zeros(1, dtype=np.float64)
        self.c_ll_p = np.zeros(1, dtype=np.float64)
        self.c_m_a = np.zeros(1, dtype=np.float64)
        self.c_m_p = np.zeros(1, dtype=np.float64)
        self.c_m_q = np.zeros(1, dtype=np.float64)
        self.c_m_r = np.zeros(1, dtype=np.float64)
        self.c_ln_b = np.zeros(1, dtype=np.float64)
        self.c_ln_p = np.zeros(1, dtype=np.float64)
        self.c_ln_q = np.zeros(1, dtype=np.float64)
        self.c_ln_r = np.zeros(1, dtype=np.float64)

        self.machs = np.zeros(1, dtype=np.float64)
        self.scale = np.ones(6, dtype=np.float64)

        self.wind_alt = np.zeros(1, dtype=np.float64)
        self.wind_speed = np.eye(2, dtype=np.float64)

        self.x_cp = np.zeros(3, dtype=np.float64)

        self.lref = 0.
        self.sref = 0.

    def rhs(self, Core):
        # wind interpolation bit
        alt = Core.lla[2]


        _VN, _VE = interp_one(alt, self.wind_alt, self.wind_speed)

        Vgust_b = np.dot(Core.DCMbe, np.array([_VN, _VE, 0.]))
        Vb = Core.DCMbc @ Core.vel

        Va = Vgust_b + Vb
        Va_norm = np.linalg.norm(Va)

        # compute stuff for aero
        mach = Va_norm/Core.a
        q = 0.5 * Core.rho * Va_norm ** 2

        alpha = np.arctan2(Va[2], Va[0])

        if Va_norm == 0.:
            beta = 0.
        else:
            beta = np.arcsin(Va[1]/Va_norm)

        alpha_raw, beta_raw = alpha, beta
        if alpha > MAX_ALPHA:
            alpha = MAX_ALPHA
        if alpha < -MAX_ALPHA:
            alpha = -MAX_ALPHA

        if beta > MAX_ALPHA:
            beta = MAX_ALPHA
        if beta < -MAX_ALPHA:
            beta = -MAX_ALPHA


        # compute p_hat, q_hat and r_hat
        if np.sqrt(Va_norm**2) == 0.:
            p_hat = 0.
            q_hat = 0.
            r_hat = 0.
        else:
            p_hat = self.lref*Core.omega[0] / (2*Va_norm)
            q_hat = self.lref*Core.omega[1] / (2*Va_norm)
            r_hat = self.lref*Core.omega[2] / (2*Va_norm)
        
        # make sure mach number is within the bounds of interpolation
        if mach < self.machs[0]:
            mach = self.machs[0]
        if mach > self.machs[-1]:
            mach = self.machs[-1]
        # interpolate the coefficients
        c_x_0 = interp_one(mach, self.machs, self.c_x_0)
        c_x_a = interp_one(mach, self.machs, self.c_x_a)
        c_x_a2 = interp_one(mach, self.machs, self.c_x_a2)
        c_x_p = interp_one(mach, self.machs, self.c_x_p)
        c_x_q = interp_one(mach, self.machs, self.c_x_q)
        c_x_r = interp_one(mach, self.machs, self.c_x_r)
        c_y_b = interp_one(mach, self.machs, self.c_y_b)
        c_y_p = interp_one(mach, self.machs, self.c_y_p)
        c_y_q = interp_one(mach, self.machs, self.c_y_q)
        c_y_r = interp_one(mach, self.machs, self.c_y_r)
        c_z_a = interp_one(mach, self.machs, self.c_z_a)
        c_z_p = interp_one(mach, self.machs, self.c_z_p)
        c_z_q = interp_one(mach, self.machs, self.c_z_q)
        c_z_r = interp_one(mach, self.machs, self.c_z_r)
        c_ll_0 = interp_one(mach, self.machs, self.c_ll_0)
        c_ll_p = interp_one(mach, self.machs, self.c_ll_p)
        c_m_a = interp_one(mach, self.machs, self.c_m_a)
        c_m_p = interp_one(mach, self.machs, self.c_m_p)
        c_m_q = interp_one(mach, self.machs, self.c_m_q)
        c_m_r = interp_one(mach, self.machs, self.c_m_r)
        c_ln_b = interp_one(mach, self.machs, self.c_ln_b)
        c_ln_p = interp_one(mach, self.machs, self.c_ln_p)
        c_ln_q = interp_one(mach, self.machs, self.c_ln_q)
        c_ln_r = interp_one(mach, self.machs, self.c_ln_r)
        x_cp = interp_one(mach, self.machs, self.x_cp)

        # compute the coefficient along axis
        c_x = c_x_0 + c_x_a2 * alpha**2 + c_x_p * p_hat + c_x_q * q_hat + c_x_r * r_hat
        # c_x = c_x_0 + c_x_a * alpha + c_x_a2 * alpha**2 + c_x_p * p_hat + c_x_q * q_hat + c_x_r * r_hat
        c_y = c_y_b * beta + c_y_p * p_hat + c_y_q * q_hat + c_y_r * r_hat
        c_z = c_z_a * alpha + c_z_p * p_hat + c_z_q * q_hat + c_z_r * r_hat

        c_ll = c_ll_0 + c_ll_p * p_hat
        c_m = c_m_a * alpha + c_m_p * p_hat + c_m_q * q_hat + c_m_r * r_hat
        c_ln = c_ln_b * beta + c_ln_p * p_hat + c_ln_q * q_hat + c_ln_r * r_hat

        # compute force and moment coefficients
        c_f = np.array([c_x, c_y, c_z])

        length = Core.cg

        c_m = np.array([c_ll, c_m, c_ln]) + np.cross(-1*length, c_f) / self.lref

        # compute actual forces and moments
        # self.scale = np.ones(6)*1.3
        aero_force = q * c_f * self.sref * self.scale[:3]

        aero_moment = q * c_m * self.sref * self.lref * self.scale[3:]

        # print(Va)
        # print(alpha, beta)
        # print(aero_force, aero_moment)
        Core.force[0] += aero_force[0]
        Core.moment[0] += aero_moment[0]
        Core.force[1] += aero_force[1]
        Core.moment[1] += aero_moment[1]
        Core.force[2] += aero_force[2]
        Core.moment[2] += aero_moment[2]

        if Core.logging:
            self.history["f_a[0]"] = np.append(self.history["f_a[0]"], aero_force[0])
            self.history["f_a[1]"] = np.append(self.history["f_a[1]"], aero_force[1])
            self.history["f_a[2]"] = np.append(self.history["f_a[2]"], aero_force[2])

            self.history["m_a[0]"] = np.append(self.history["m_a[0]"], aero_moment[0])
            self.history["m_a[1]"] = np.append(self.history["m_a[1]"], aero_moment[1])
            self.history["m_a[2]"] = np.append(self.history["m_a[2]"], aero_moment[2])

            self.history["M"] = np.append(self.history["M"], mach)
            self.history["q"] = np.append(self.history["q"], q)

            self.history["alpha"] = np.append(self.history["alpha"], alpha_raw)
            self.history["beta"] = np.append(self.history["beta"], beta_raw)

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["M"] = np.zeros(1, dtype=np.float64)
        self.history["q"] = np.zeros(1, dtype=np.float64)
        self.history["alpha"] = np.zeros(1, dtype=np.float64)
        self.history["beta"] = np.zeros(1, dtype=np.float64)

        self.history["f_a[0]"] = np.zeros(1, dtype=np.float64)
        self.history["f_a[1]"] = np.zeros(1, dtype=np.float64)
        self.history["f_a[2]"] = np.zeros(1, dtype=np.float64)

        self.history["m_a[0]"] = np.zeros(1, dtype=np.float64)
        self.history["m_a[1]"] = np.zeros(1, dtype=np.float64)
        self.history["m_a[2]"] = np.zeros(1, dtype=np.float64)
        # print(mach)
        # print(Core.moment)



aeroconstant_spec = empty_spec + \
    [("c_x_0", nb.float64),
     ("c_x_a2", nb.float64),
     ("c_x_p", nb.float64),
     ("c_x_q", nb.float64),
     ("c_x_r", nb.float64),
     ("c_y_b", nb.float64),
     ("c_y_p", nb.float64),
     ("c_y_q", nb.float64),
     ("c_y_r", nb.float64),
     ("c_z_a", nb.float64),
     ("c_z_p", nb.float64),
     ("c_z_q", nb.float64),
     ("c_z_r", nb.float64),
     ("c_ll_0", nb.float64),
     ("c_ll_p", nb.float64),
     ("c_m_a", nb.float64),
     ("c_m_p", nb.float64),
     ("c_m_q", nb.float64),
     ("c_m_r", nb.float64),
     ("c_ln_b", nb.float64),
     ("c_ln_p", nb.float64),
     ("c_ln_q", nb.float64),
     ("c_ln_r", nb.float64),

     ("machs", nb.float64),
     ("scale", nb.float64[:]),

     ("wind_alt", nb.float64),
     ("wind_heading", nb.float64),
     ("wind_speed", nb.float64),

     ("bref", nb.float64),
     ("cref", nb.float64),
     ("sref", nb.float64)
     ]

@nb.jitclass(aeroconstant_spec)
class AeroConstant(object):
    def __init__(self):
        self.id = 'AeroConstant'
        self.type = 'Aero'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.c_x_0 = 0.0
        self.c_x_a2 = 0.0
        self.c_x_p = 0.0
        self.c_x_q = 0.0
        self.c_x_r = 0.0
        self.c_y_b = 0.0
        self.c_y_p = 0.0
        self.c_y_q = 0.0
        self.c_y_r = 0.0
        self.c_z_a = 0.0
        self.c_z_p = 0.0
        self.c_z_q = 0.0
        self.c_z_r = 0.0
        self.c_ll_0 = 0.0
        self.c_ll_p = 0.0
        self.c_m_a = 0.0
        self.c_m_p = 0.0
        self.c_m_q = 0.0
        self.c_m_r = 0.0
        self.c_ln_b = 0.0
        self.c_ln_p = 0.0
        self.c_ln_q = 0.0
        self.c_ln_r = 0.0

        self.machs = 0.0
        self.scale = np.ones(3, dtype=np.float64)

        self.wind_alt = 0.0
        self.wind_heading = 0.0
        self.wind_speed = 0.0

        self.bref = 0.0
        self.cref = 0.0
        self.sref = 0.0

    def rhs(self, Core):
        lref = self.bref
        # wind interpolation bit
        alt = Core.lla[2]

        wind_v = self.wind_speed
        wind_angle = self.wind_heading

        wind_angle = wind_angle/180*np.pi

        # find the aerodynamic frame velocity

        _VN = wind_v*np.cos(wind_angle)
        _VE = wind_v*np.sin(wind_angle)
        Vgust_b = np.dot(Core.DCMbe, np.array([_VN, _VE, 0.]))

        # Vb = Core.DCMbc @ Core.DCMci @ Core.vel
        Vb = Core.DCMbc @ Core.vel
        Va = Vgust_b + Vb
        Va_norm = np.linalg.norm(Va)

        # compute stuff for aero
        mach = Va_norm/Core.a
        q = 0.5 * Core.rho * Va_norm ** 2

        alpha = np.arctan2(Va[2], Va[0])

        if Va_norm == 0.:
            beta = 0.
        else:
            beta = np.arcsin(Va[1]/Va_norm)

        if alpha > MAX_ALPHA:
            alpha = MAX_ALPHA
        if alpha < -MAX_ALPHA:
            alpha = -MAX_ALPHA

        # compute p_hat, q_hat and r_hat
        if np.sqrt(Va_norm**2) <= 1e-8:
            p_hat = 0.
            q_hat = 0.
            r_hat = 0.
        else:
            p_hat = self.bref*Core.omega[0] / (2*Va_norm)
            q_hat = self.cref*Core.omega[1] / (2*Va_norm)
            r_hat = self.bref*Core.omega[2] / (2*Va_norm)

        # compute the coefficient along axis
        c_x = self.c_x_0 + self.c_x_a2 * alpha**2 + self.c_x_p * p_hat + self.c_x_q * q_hat + self.c_x_r * r_hat
        c_y = self.c_y_b * beta + self.c_y_p * p_hat + self.c_y_q * q_hat + self.c_y_r * r_hat
        c_z = self.c_z_a * alpha + self.c_z_p * p_hat + self.c_z_q * q_hat + self.c_z_r * r_hat

        c_ll = self.c_ll_0 + self.c_ll_p * p_hat
        c_m = self.c_m_a * alpha + self.c_m_p * p_hat + self.c_m_q * q_hat + self.c_m_r * r_hat
        c_ln = self.c_ln_b * beta + self.c_ln_p * p_hat + self.c_ln_q * q_hat + self.c_ln_r * r_hat

        # compute force and moment coefficients
        c_f = np.array([c_x, c_y, c_z])

        # l = -Core.cg+np.array([x_cp, 0, 0])
        length = Core.cg
        c_m = np.array([c_ll, c_m, c_ln]) + np.cross(length, c_f) / np.array([self.bref, self.cref, self.bref])

        # compute actual forces and moments
        aero_force = q * c_f * self.sref * self.scale
        aero_force_c = aero_force

        aero_moment = q * c_m * self.sref * np.array([self.bref, self.cref, self.bref]) * self.scale

        Core.force[0] += aero_force_c[0]
        Core.moment[0] += aero_moment[0]
        Core.force[1] += aero_force_c[1]
        Core.moment[1] += aero_moment[1]
        Core.force[2] += aero_force_c[2]
        Core.moment[2] += aero_moment[2]

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)
