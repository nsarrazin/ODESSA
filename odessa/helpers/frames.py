import numpy as np
import numba as nb
import math


@nb.jit(nopython=True, cache=True)
def wgs84_to_ecef(lla):
    """Converts Longitude/Latitude/Altitude to ECEF coordinates.

    Args:
        lla ([floats]): A list of floats representing LLA coordinates in radians.

    Returns:
        [float] : A list of floats representing the ECEF coordinates [m]
    """
    coords_ecef = np.zeros(3, dtype=np.float64)

    a = 6378137.0
    b = 6356752.314245179
    ba = 0.9966471893352525

    psi = np.arctan(np.tan(lla[0]) * ba)
    r = a * np.cos(psi) + lla[2] * np.cos(lla[0])

    coords_ecef[0] = r * np.cos(lla[1])
    coords_ecef[1] = r * np.sin(lla[1])
    coords_ecef[2] = b * np.sin(psi) + lla[2] * np.sin(lla[0])

    return coords_ecef


@nb.jit(nopython=True, cache=True)
def ecef_to_wgs84(x_ecef):
    x = x_ecef[0]
    y = x_ecef[1]
    z = x_ecef[2]

    a = 6378137.
    e = 8.1819190842622e-2

    asq = np.power(a, 2)
    esq = np.power(e, 2)
    b = np.sqrt(asq * (1-esq))
    bsq = np.power(b, 2)

    ep = np.sqrt((asq - bsq)/bsq)

    p = np.sqrt(np.power(x, 2) + np.power(y, 2))

    th = np.arctan2(a*z, b*p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + np.power(ep, 2)*b*np.power(np.sin(th), 3)),
                     (p - esq*a*np.power(np.cos(th), 3)))

    N = a/(np.sqrt(1-esq*np.power(np.sin(lat), 2)))

    alt = p / np.cos(lat) - N
    results = np.zeros(3)
    results = np.array([lat, lon, alt], dtype=np.float64)
    return results


@nb.jit(nopython=True, cache=True)
def vehicle_carried_to_ecef(lla, vector):
    T_e = np.zeros((3, 3), dtype=np.float64)
    # print(lla)
    T_e = np.array([[-np.sin(lla[0])*np.cos(lla[1]), -np.sin(lla[0])*np.sin(lla[1]),  np.cos(lla[0])],
                    [-np.sin(lla[1]),  np.cos(lla[1]),  0],
                    [-np.cos(lla[0])*np.cos(lla[1]), -np.cos(lla[0])*np.sin(lla[1]), -np.sin(lla[0])]])

    return T_e @ vector


@nb.jit(nopython=True, cache=True)
def hgeodet_to_hgeopot(h_geodet):
    Re = 6371000.8
    return Re * h_geodet / (Re + h_geodet)


@nb.jit(nopython=True, cache=True)
def hgeopot_to_hgeodet(h_geopot):
    Re = 6371000.8
    return Re * h_geopot / (Re - h_geopot)


@nb.jit(nopython=True, cache=True)
def ecef_to_rotating_lla(x_inertial, w_e, t):
    _c = np.cos(w_e*t)
    _s = np.sin(w_e*t)
    matrix = np.zeros(3)
    matrix = np.array([x_inertial[0] * _c + x_inertial[1] * _s,
                       -x_inertial[0] * _s + x_inertial[1] * _c,
                       x_inertial[2]])
    vals = np.zeros(3)
    vals = ecef_to_wgs84(matrix)
    return vals


@nb.njit
def ecef_to_spherical(Xc):
    x = Xc[0, 0]
    y = Xc[1, 0]
    z = Xc[2, 0]
    r = math.sqrt(x**2+y**2+z**2)
    lat_gc = math.asin(z/r)
    lon_gc = math.atan2(y, x)

    return [r, lat_gc, lon_gc]
