import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type
from ....helpers.math_func import interp_one

aerolinear_spec = empty_spec + \
    [("c_x_0", nb.float64[:]),

     ("machs", nb.float64[:]),
     ("scale", nb.float64[:]),

     ("wind_alt", nb.float64[:]),
     ("wind_heading", nb.float64[:]),
     ("wind_speed", nb.float64[:]),

     ("x_cp", nb.float64[:]),
     ("lref", nb.float64),
     ("sref", nb.float64)
     ]

@nb.jitclass(aerolinear_spec)
class AeroBallistic6DoF(object):
    """The Ballistic aerodynamics module for 6DoF applications.

       Drag is applied in the opposite direction to the velocity vector.
       Wind is included in the computation.
    """
    def __init__(self):
        self.id = 'AeroBallistic6DoF'
        self.type = 'Aero'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.c_x_0 = np.zeros(1, dtype=np.float64)

        self.machs = np.zeros(1, dtype=np.float64)
        self.scale = np.ones(6, dtype=np.float64)

        self.wind_alt = np.zeros(1, dtype=np.float64)
        self.wind_heading = np.zeros(1, dtype=np.float64)
        self.wind_speed = np.zeros(1, dtype=np.float64)

        self.lref = 0.
        self.sref = 0.

    def rhs(self, Core):
        # wind interpolation bit
        alt = Core.lla[2]

        wind_v = interp_one(alt, self.wind_alt, self.wind_speed)
        wind_angle = interp_one(alt, self.wind_alt, self.wind_heading)

        wind_angle = wind_angle/180*np.pi

        # find the aerodynamic frame velocity

        _VN = wind_v*np.cos(wind_angle)
        _VE = wind_v*np.sin(wind_angle)
        Vgust_b = np.dot(Core.DCMbe, np.array([_VN, _VE, 0.]))
        Vb = Core.DCMbc @ Core.vel

        Va = Vgust_b + Vb
        Va_norm = np.linalg.norm(Va)

        # compute stuff for aero
        mach = Va_norm/Core.a

        # make sure mach number is within the bounds of interpolation
        if mach < self.machs[0]:
            mach = self.machs[0]
        if mach > self.machs[-1]:
            mach = self.machs[-1]
        # interpolate the coefficients
        c_x_0 = interp_one(mach, self.machs, self.c_x_0)

        # compute the coefficient along axis
        c_x = c_x_0 

        F_drag = 0.5*Core.rho*c_x_0*self.sref*Va_norm**2
        
        if Va_norm > 0:
            aero_force = [F_drag*Va[0]/Va_norm,
                          F_drag*Va[1]/Va_norm,
                          F_drag*Va[2]/Va_norm]
        else:
            aero_force = [0.,0.,0.]
        
        Core.force[0] += aero_force[0]
        Core.force[1] += aero_force[1]
        Core.force[2] += aero_force[2]

        if Core.logging:
            self.history["f_a[0]"] = np.append(self.history["f_a[0]"], aero_force[0])
            self.history["f_a[1]"] = np.append(self.history["f_a[1]"], aero_force[1])
            self.history["f_a[2]"] = np.append(self.history["f_a[2]"], aero_force[2])

            # self.history["m_a[0]"] = np.append(self.history["m_a[0]"], aero_moment[0])
            # self.history["m_a[1]"] = np.append(self.history["m_a[1]"], aero_moment[1])
            # self.history["m_a[2]"] = np.append(self.history["m_a[2]"], aero_moment[2])

            self.history["M"] = np.append(self.history["M"], mach)
            # self.history["q"] = np.append(self.history["q"], q)

            # self.history["alpha"] = np.append(self.history["alpha"], alpha_raw)
            # self.history["beta"] = np.append(self.history["beta"], beta_raw)

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