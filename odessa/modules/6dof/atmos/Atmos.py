import numba as nb
import numpy as np
from ...Empty import empty_spec, float_array_type

from ....helpers.atmos import atmo_isa
from ....helpers.frames import hgeodet_to_hgeopot
from ....helpers.math_func import interp_one

AtmosISA_spec = empty_spec + []


@nb.jitclass(AtmosISA_spec)
class AtmosISA(object):
    """The Atmospherics module using the ISA atmospheric model.
    """
    def __init__(self):
        self.id = 'AtmosISA'
        self.type = 'Atmos'

        self.init_history()

    def rhs(self, Core):
        alt = hgeodet_to_hgeopot(Core.lla[2])

        T, p, rho, a = atmo_isa(alt)

        Core.rho = rho
        Core.a = a
        Core.p = p
        
        if Core.logging:
            self.history["T"] = np.append(self.history["T"], T)
            self.history["p"] = np.append(self.history["p"], p)
            self.history["rho"] = np.append(self.history["rho"], rho)
            self.history["a"] = np.append(self.history["a"], a)

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

        self.history["T"] = np.zeros(1, dtype=np.float64)
        self.history["p"] = np.zeros(1, dtype=np.float64)
        self.history["rho"] = np.zeros(1, dtype=np.float64)
        self.history["a"] = np.zeros(1, dtype=np.float64)


@nb.jitclass(AtmosISA_spec)
class AtmosConstant(object):
    """The Atmos module with constant atmospheric properties.
    """
    def __init__(self):
        self.id = 'AtmosConstant'
        self.type = 'Atmos'
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)

    def rhs(self, Core):
        Core.rho = 1.225
        Core.a = 343

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)


@nb.jitclass(AtmosISA_spec)
class AtmosSTD76(object):
    """The Atmospherics module using the STD76 atmospheric model.
    """

    def __init__(self):
        self.id = "AtmosSTD76"
        self.type = 'Atmos'

    def rhs(self, Core):
        table = np.array([[0.00000, 1.22500, 340.294], [1.00000, 1.11164, 336.434], [2.00000, 1.00649, 332.529], [3.00000, 0.909122, 328.578], [4.00000, 0.819129, 324.579], [5.00000, 0.736116, 320.529], [6.00000, 0.659697, 316.428], [7.00000, 0.589501, 312.274], [8.00000, 0.525168, 308.063], [9.00000, 0.466348, 303.793], [10.0000, 0.412707, 299.463], [11.0000, 0.363918, 295.070], [12.0000, 0.310828, 295.070], [13.0000, 0.265483, 295.070], [14.0000, 0.226753, 295.070], [15.0000, 0.193674, 295.070], [16.0000, 0.165420, 295.070], [17.0000, 0.141288, 295.070], [18.0000, 0.120676, 295.070], [19.0000, 0.103071, 295.070], [20.0000, 0.0880349, 295.070], [21.0000, 0.0748737, 295.750], [22.0000, 0.0637273, 296.428], [23.0000, 0.0542803, 297.105], [24.0000, 0.0462674, 297.781], [25.0000, 0.0394658, 298.455], [26.0000, 0.0336882, 299.128], [27.0000, 0.0287769, 299.799], [28.0000, 0.0245988, 300.468], [29.0000, 0.0210420, 301.136], [30.0000, 0.0180119, 301.803], [31.0000, 0.0154288, 302.468], [32.0000, 0.0132250, 303.131], [33.0000, 0.0112620, 304.982], [34.0000, 0.00960889, 306.821], [35.0000, 0.00821392, 308.649], [36.0000, 0.00703441, 310.467], [37.0000, 0.00603513, 312.274], [38.0000, 0.00518691, 314.070], [39.0000, 0.00446557, 315.856], [40.0000, 0.00385101, 317.633], [41.0000, 0.00332648, 319.399], [42.0000, 0.00287802, 321.156], [43.0000, 0.00249393, 322.903], [44.0000, 0.00216443, 324.641],
                          [45.0000, 0.00188129, 326.369], [46.0000, 0.00163760, 328.088], [47.0000, 0.00142753, 329.799], [48.0000, 0.00125825, 329.799], [49.0000, 0.00110904, 329.799], [50.0000, 0.000977525, 329.799], [51.0000, 0.000861606, 329.799], [52.0000, 0.000766867, 328.088], [53.0000, 0.000681710, 326.369], [54.0000, 0.000605252, 324.641], [55.0000, 0.000536684, 322.903], [56.0000, 0.000475263, 321.156], [57.0000, 0.000420311, 319.399], [58.0000, 0.000371207, 317.633], [59.0000, 0.000327382, 315.856], [60.0000, 0.000288321, 314.070], [61.0000, 0.000253550, 312.274], [62.0000, 0.000222640, 310.467], [63.0000, 0.000195200, 308.649], [64.0000, 0.000170875, 306.821], [65.0000, 0.000149342, 304.982], [66.0000, 0.000130308, 303.131], [67.0000, 0.000113510, 301.269], [68.0000, 0.0000987069, 299.396], [69.0000, 0.0000856830, 297.511], [70.0000, 0.0000742430, 295.614], [71.0000, 0.0000642110, 293.704], [72.0000, 0.0000552370, 292.333], [73.0000, 0.0000474496, 290.955], [74.0000, 0.0000407010, 289.570], [75.0000, 0.0000348607, 288.179], [76.0000, 0.0000298135, 286.781], [77.0000, 0.0000254579, 285.377], [78.0000, 0.0000217046, 283.965], [79.0000, 0.0000184751, 282.546], [80.0000, 0.0000157005, 281.120], [81.0000, 0.0000133205, 279.687], [82.0000, 0.0000112820, 278.246], [83.0000, 0.00000953899, 276.798], [84.0000, 0.00000805098, 275.343], [85.0000, 0.00000677222, 274.096], [86.0000, 0.00000564114, 274.096],
                          [86.0000, 0., 274.096], [1e6, 0., 274.096]])
        alt = hgeodet_to_hgeopot(Core.lla[2])

        alts = table[:, 0]*1000
        rhos = table[:, 1]
        sounds = table[:, 2]

        rho = interp_one(alt, alts, rhos)
        a = interp_one(alt, alts, sounds)

        # if alt > 8.6e4:
        #     rho = 0.

        Core.rho = rho
        Core.a = a

    def init_history(self):
        self.history = nb.typed.Dict.empty(key_type=nb.types.unicode_type,
                                           value_type=float_array_type)
