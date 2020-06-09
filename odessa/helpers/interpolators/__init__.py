from .Interpolator1DLinear import Interpolator1DLinear
from .RegularGridInterpolator import RegularGridInterpolator

interpolators = {"INTERP1DLINEAR" : Interpolator1DLinear,
                 "REGULARGRID": RegularGridInterpolator}