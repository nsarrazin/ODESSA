"""Module that contains a number of functions to create transformation matrices between different reference frames."""
import numpy as np
import math
from numpy import ndarray as numpy_array
import numba as nb


@nb.njit(cache=True)
def TCI(t: float, omega_t: float = 7.2921235169904*10**(-5)) -> numpy_array:
    """Transformation Matrix from :ref:s`sec:F-I` to :ref:`sec:F-C`

    :param t: Time (Seconds)
    :type t: float
    :param omega_t: Earth's rotational rate (radians/s)
    :type omega_t: float
    :return: TCI
    :rtype: numpy_array
    """
    return np.array([[math.cos(omega_t*t),  math.sin(omega_t*t), 0.],
                     [-math.sin(omega_t*t), math.cos(omega_t*t), 0.],
                     [0.,                    0.,                   1]])


@nb.njit(cache=True)
def TIC(t: float, omega_t: float = 7.2921235169904e-5) -> numpy_array:
    """Transformation Matrix from :ref:`sec:F-C` to  :ref:`sec:F-I`

    :param t: Time (Seconds)
    :type t: float
    :param omega_t: Earth's rotational rate (radians/s)
    :type omega_t: float
    :return: TIC
    :rtype: numpy_array
    """
    return TCI(t, omega_t).transpose()


@nb.njit(cache=True)
def Tab(alpha: float, beta: float) -> numpy_array:
    """Transformation Matrix from :ref:`sec:F-b` to :ref:`sec:F-a`

    :param alpha: Aerodynamic angle of attack (radians)
    :type alpha: float
    :param beta: Aerodynamic angle of side-slip (radians)
    :type beta: float
    :return: Tab
    :rtype: numpy_array"""

    sang = np.sin(np.array([alpha, beta]))
    cang = np.cos(np.array([alpha, beta]))

    return np.array([[cang[1]*cang[0], sang[1], cang[1]*sang[0]],
                     [-sang[1]*cang[0], cang[1], -sang[1]*sang[0]],
                     [-sang[0], 0., cang[0]]])


@nb.njit(cache=True)
def Tba(alpha: float, beta: float) -> numpy_array:
    """Transformation Matrix from :ref:`sec:F-a` to :ref:`sec:F-b`

    :param alpha: Aerodynamic angle of attack (radians)
    :type alpha: float
    :param beta: Aerodynamic angle of side-slip (radians)
    :type beta: float
    :return: Tba
    :rtype: numpy_array"""

    return Tab(alpha, beta).transpose()


@nb.njit(cache=True)
def TEC(tau: float, delta: float) -> numpy_array:
    """Transformation from :ref:`sec:F-C` to the :ref:`sec:F-E`

    :param tau: Longitude (radians) from the Greenwich meridian (tau is positive if the vehicle position is east of the Greenwich meridian)
    :type tau: float
    :param delta: Latitude (radians) from the equator (delta is positive if the vehicle location is on the northern hemisphere)
    :type delta: float
    :return: TEC
    :rtype: numpy_array"""

    sang = np.sin(np.array([tau, delta]))
    cang = np.cos(np.array([tau, delta]))

    return np.array([[-sang[1]*cang[0], -sang[1]*sang[0], cang[1]],
                     [-sang[0], cang[0], 0.],
                     [-cang[1]*cang[0], -cang[1]*sang[0], -sang[1]]])


@nb.njit(cache=True)
def TCE(tau: float, delta: float) -> numpy_array:
    """Transformation from :ref:`sec:F-E` to the :ref:`sec:F-C`

    :param tau: Longitude (radians) from the Greenwich meridian (tau is positive if the vehicle position is east of the Greenwich meridian)
    :type tau: float
    :param delta: Latitude (radians) from the equator (delta is positive if the vehicle location is on the northern hemisphere)
    :type delta: float
    :return: TEC
    :rtype: numpy_array"""

    return TEC(tau, delta).transpose()


@nb.njit(cache=True)
def TbE(psi: float, theta: float, phi: float) -> numpy_array:
    """Transformation from :ref:`sec:F-E` to the :ref:`sec:F-b`

    :param psi: Yaw angle about the Z_E-axis (radians)
    :type psi: float
    :param theta: Pitch angle about the Y_E-axis (radians)
    :type theta: float
    :param psi: Roll angle about the X_E-axis (radians)
    :type psi: float
    :return: TbE
    :rtype: numpy_array"""

    sang = np.sin(np.array([psi, theta, phi]))
    cang = np.cos(np.array([psi, theta, phi]))

    return np.array([[cang[1]*cang[0], cang[1]*sang[0], -sang[1]],
                     [sang[2]*sang[1]*cang[0]-cang[2]*sang[0], sang[2] *
                         sang[1]*sang[0]+cang[2]*cang[0], sang[2]*cang[1]],
                     [cang[2]*sang[1]*cang[0]+sang[2]*sang[0], cang[2]*sang[1]*sang[0]-sang[2]*cang[0], cang[2]*cang[1]]])


@nb.njit(cache=True)
def TEb(psi: float, theta: float, phi: float) -> numpy_array:
    """Transformation from the :ref:`sec:F-b` to :ref:`sec:F-E`

    :param psi: Yaw angle about the Z_E-axis (radians)
    :type psi: float
    :param theta: Pitch angle about the Y_E-axis (radians)
    :type theta: float
    :param psi: Roll angle about the Z_E-axis (radians)
    :type psi: float
    :return: TbE
    :rtype: numpy_array"""

    return TbE(psi, theta, phi).transpose()

@nb.njit(cache=True)
def TaE(chi: float, gamma: float, mu: float) -> numpy_array:
    """Transformation from the :ref:`sec:F-b` to the :ref:`sec:F-a`

    :param chi: Aerodynamic heading angle about the Z_E-axis (radians)
    :type chi: float
    :param gamma: Aerodynamic flight-path angle about the Y_E-axis (radians)
    :type gamma: float
    :param mu: Aerodynamic bank angle about the X_a-axis (radians)
    :type mu: float
    :return: TaE
    :rtype: numpy_array"""

    sang = np.sin(np.array([chi, gamma, mu]))
    cang = np.cos(np.array([chi, gamma, mu]))

    return np.array([[cang[1]*cang[0], cang[1]*sang[0], -sang[1]],
                     [-sang[2]*sang[1]*cang[0]-cang[2]*sang[0], -sang[2] *
                         sang[1]*sang[0]+cang[2]*cang[0], -sang[2]*cang[1]],
                     [cang[2]*sang[1]*cang[0]-sang[2]*sang[0], cang[2]*sang[1]*sang[0]+sang[2]*cang[0], cang[2]*cang[1]]])


@nb.njit(cache=True)
def TEa(chi: float, gamma: float, mu: float) -> numpy_array:
    """Transformation from the :ref:`sec:F-a` to the :ref:`sec:F-b`

    :param chi: Aerodynamic heading angle about the Z_E-axis (radians)
    :type chi: float
    :param gamma: Aerodynamic flight-path angle about the Y_E-axis (radians)
    :type gamma: float
    :param mu: Aerodynamic bank angle about the X_a-axis (radians)
    :type mu: float
    :return: TaE
    :rtype: numpy_array"""

    return TaE(chi, gamma, mu).transpose()


@nb.njit(cache=True)
def ecef2spherical(Xc: numpy_array):
    x = float(Xc[0])
    y = float(Xc[1])
    z = float(Xc[2])
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat_gc = math.asin(z / r)
    lon_gc = math.atan2(y, x)

    return r, lat_gc, lon_gc
