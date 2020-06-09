"""Module that contains functions to deal 
with rotation in different forms.
(DCM, euler angles, and quaternions)"""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def _threeaxisrot(r11, r12, r21, r31, r32):
    r0 = np.arctan2(r11, r12)
    r1 = np.arcsin(r21)
    r2 = np.arctan2(r31, r32)
    return r0, r1, r2


@nb.njit(cache=True)
def _twoaxisrot(r11, r12, r21, r31, r32):
    r0 = np.arctan2(r11, r12)
    r1 = np.arccos(r21)
    r2 = np.arctan2(r31, r32)
    return r0, r1, r2


@nb.njit(cache=True)
def dcm2angles(DCM, rtype='zyx', unit='rad'):
    """Converts Direction Cosine Matrix to Euler angles.

    Parameters
    ----------
    DCM : array_like
        Direction Cosine Matrix.
    rtype : string, optional
        Type of Euler angle to convert to (the default is zyx).
    unit : string
        Unit of the angles, can be 'deg' or 'rad' (default is rad).

    Returns
    -------
    r0 : float
        First rotation angle.
    r1 : float
        Second rotation angle.
    r2 : float
        Third rotation angle.
    """

    # rtype = rtype.lower()

    r0 = 0
    r1 = 0
    r2 = 0

    if rtype == 'zyx':
        r0, r1, r2 = _threeaxisrot(
            DCM[0, 1], DCM[0, 0], -(DCM[0, 2]), DCM[1, 2], DCM[2, 2])
    elif rtype == 'zyz':
        r0, r1, r2 = _twoaxisrot(
            DCM[2, 1], DCM[2, 0], DCM[2, 2], DCM[1, 2], -DCM[0, 2])
    elif rtype == 'zxy':
        r0, r1, r2 = _threeaxisrot(-DCM[1, 0], DCM[1, 1],
                                   DCM[1, 2], -DCM[0, 2], DCM[2, 2])
    elif rtype == 'zxz':
        r0, r1, r2 = _twoaxisrot(
            DCM[2, 0], -DCM[2, 1], DCM[2, 2], DCM[0, 2], DCM[1, 2])
    elif rtype == 'yxz':
        r0, r1, r2 = _threeaxisrot(
            DCM[2, 0], DCM[2, 2], -DCM[2, 1], DCM[0, 1], DCM[1, 1])
    elif rtype == 'yxy':
        r0, r1, r2 = _twoaxisrot(
            DCM[1, 0], DCM[1, 2], DCM[1, 1], DCM[0, 1], -DCM[2, 1])
    elif rtype == 'yzx':
        r0, r1, r2 = _threeaxisrot(-DCM[0, 2], DCM[0, 0],
                                   DCM[0, 1], -DCM[2, 1], DCM[1, 1])
    elif rtype == 'yzy':
        r0, r1, r2 = _twoaxisrot(
            DCM[1, 2], -DCM[1, 0], DCM[1, 1], DCM[2, 1], DCM[0, 1])
    elif rtype == 'xyz':
        r0, r1, r2 = _threeaxisrot(-DCM[2, 1], DCM[2, 2],
                                   DCM[2, 0], -DCM[1, 0], DCM[0, 0])
    elif rtype == 'xyx':
        r0, r1, r2 = _twoaxisrot(
            DCM[0, 1], -DCM[0, 2], DCM[0, 0], DCM[1, 0], DCM[2, 0])
    elif rtype == 'xzy':
        r0, r1, r2 = _threeaxisrot(
            DCM[1, 2], DCM[1, 1], -DCM[1, 0], DCM[2, 0], DCM[0, 0])
    elif rtype == 'xzx':
        r0, r1, r2 = _twoaxisrot(
            DCM[0, 2], DCM[0, 1], DCM[0, 0], DCM[2, 0], -DCM[1, 0])

    if unit == 'rad':
        return np.array([r0, r1, r2])
    else:
        return np.array([np.rad2deg(r0), np.rad2deg(r1), np.rad2deg(r2)])


@nb.njit(cache=True)
def dcm2quats(DCM):
    q = np.array([[1.], [1.], [1.], [1.]])

    r_a = np.array([1 + DCM[0, 0] + DCM[1, 1] + DCM[2, 2],
                    1 + DCM[0, 0] - DCM[1, 1] - DCM[2, 2],
                    1 - DCM[0, 0] + DCM[1, 1] - DCM[2, 2],
                    1 - DCM[0, 0] - DCM[1, 1] + DCM[2, 2]])
    case = np.argmax(r_a)

    if case == 0:
        r = np.sqrt(r_a[case])
        q[0] = 0.5 * r
        q[1] = 0.5 * (DCM[1][2]-DCM[2][1])/r
        q[2] = 0.5 * (DCM[2][0]-DCM[0][2])/r
        q[3] = 0.5 * (DCM[0][1]-DCM[1][0])/r

    elif case == 1:
        r = np.sqrt(r_a[case])
        q[0] = 0.5 * (DCM[1][2]-DCM[2][1])/r
        q[1] = 0.5 * r
        q[2] = 0.5 * (DCM[0][1]+DCM[1][0])/r
        q[3] = 0.5 * (DCM[2][0]+DCM[0][2])/r

    elif case == 2:
        r = np.sqrt(r_a[case])
        q[0] = 0.5 * (DCM[2][0] - DCM[0][2])/r
        q[1] = 0.5 * (DCM[0][1] + DCM[1][0])/r
        q[2] = 0.5 * r
        q[3] = 0.5 * (DCM[1][2] + DCM[2][1])/r

    elif case == 3:
        r = np.sqrt(r_a[case])
        q[0] = 0.5 * (DCM[0][1] - DCM[1][0])/r
        q[1] = 0.5 * (DCM[2][0] + DCM[0][2])/r
        q[2] = 0.5 * (DCM[1][2] + DCM[2][1])/r
        q[3] = 0.5 * r
    else:
        raise ValueError("Invalid DCM")
    return q.flatten()


@nb.njit(cache=True)
def angles2dcm(angles, rtype='zyx', unit='rad'):
    """Converts Euler angles to Direction Cosine Matrix.

    Parameters
    ----------
    phi : float
        First rotation angle (Yaw angle in case of "zyx").
    theta : float
        Second rotation angle (Pitch angle in case of "zyx").
    psi : float
        Third rotation angle (Roll angle in case of "zyx").
    rtype : string, optional
        Rotational order (default is zyx).
    unit : string, optional
        Unit of the returned angles, rad or deg (default is rad).

    Returns
    -------
    DCM : numpy.ndarray
        Direction Cosine Matrix.

    Raises
    ------
    ValueError
        If the passed unit is not 'deg' of 'rad'.
    ValueError
        If the passed rotational order is not supported.
    """
    if unit == 'deg':
        angles = angles / 180*np.pi
    elif unit == 'rad':
        pass
    else:
        raise ValueError('Invalid unit')

    cang = np.cos(angles)
    sang = np.sin(angles)
    # rtype = str.lower(rtype)

    if rtype == "zyx":
        return np.array([[cang[1]*cang[0], cang[1]*sang[0], -sang[1]],
                         [sang[2]*sang[1]*cang[0] - cang[2]*sang[0], sang[2]
                             * sang[1]*sang[0] + cang[2]*cang[0], sang[2]*cang[1]],
                         [cang[2]*sang[1]*cang[0] + sang[2]*sang[0], cang[2]*sang[1]*sang[0] - sang[2]*cang[0], cang[2]*cang[1]]])
    elif rtype == "zyz":
        return np.array([[cang[0]*cang[2]*cang[1] - sang[0]*sang[2], sang[0]*cang[2]*cang[1] + cang[0]*sang[2], -sang[1]*cang[2]],
                         [-cang[0]*cang[1]*sang[2] - sang[0]*cang[2], -sang[0]
                             * cang[1]*sang[2] + cang[0]*cang[2], sang[1]*sang[2]],
                         [cang[0]*sang[1], sang[0]*sang[1], cang[1]]])
    elif rtype == "zxy":
        return np.array([[cang[2]*cang[0] - sang[1]*sang[2]*sang[0], cang[2]*sang[0] + sang[1]*sang[2]*cang[0], -sang[2]*cang[1]],
                         [-cang[1]*sang[0], cang[1]*cang[0], sang[1]],
                         [sang[2]*cang[0] + sang[1]*cang[2]*sang[0], sang[2]*sang[0] - sang[1]*cang[2]*cang[0], cang[1]*cang[2]]])
    elif rtype == "zxz":
        return np.array([[-sang[0]*cang[1]*sang[2] + cang[0]*cang[2], cang[0]*cang[1]*sang[2] + sang[0]*cang[2], sang[1]*sang[2]],
                         [-sang[0]*cang[2]*cang[1] - cang[0]*sang[2], cang[0]
                             * cang[2]*cang[1] - sang[0]*sang[2], sang[1]*cang[2]],
                         [sang[0]*sang[1], -cang[0]*sang[1], cang[1]]])
    elif rtype == "yxz":
        return np.array([[cang[0]*cang[2] + sang[1]*sang[0]*sang[2], cang[1]*sang[2], -sang[0]*cang[2] + sang[1]*cang[0]*sang[2]],
                         [-cang[0]*sang[2] + sang[1]*sang[0]*cang[2], cang[1]
                             * cang[2], sang[0]*sang[2] + sang[1]*cang[0]*cang[2]],
                         [sang[0]*cang[1], -sang[1], cang[1]*cang[0]]])
    elif rtype == "yxy":
        return np.array([[-sang[0]*cang[1]*sang[2] + cang[0]*cang[2], sang[1]*sang[2], -cang[0]*cang[1]*sang[2] - sang[0]*cang[2]],
                         [sang[0]*sang[1], cang[1], cang[0]*sang[1]],
                         [sang[0]*cang[2]*cang[1] + cang[0]*sang[2], -sang[1]*cang[2], cang[0]*cang[2]*cang[1] - sang[0]*sang[2]]])
    elif rtype == "yzx":
        return np.array([[cang[0]*cang[1], sang[1], -sang[0]*cang[1]],
                         [-cang[2]*cang[0]*sang[1] + sang[2]*sang[0], cang[1]
                             * cang[2], cang[2]*sang[0]*sang[1] + sang[2]*cang[0]],
                         [sang[2]*cang[0]*sang[1] + cang[2]*sang[0], -sang[2]*cang[1], -sang[2]*sang[0]*sang[1] + cang[2]*cang[0]]])
    elif rtype == "yzy":
        return np.array([[cang[0]*cang[2]*cang[1] - sang[0]*sang[2], sang[1]*cang[2], -sang[0]*cang[2]*cang[1] - cang[0]*sang[2]],
                         [-cang[0]*sang[1], cang[1], sang[0]*sang[1]],
                         [cang[0]*cang[1]*sang[2] + sang[0]*cang[2], sang[1]*sang[2], -sang[0]*cang[1]*sang[2] + cang[0]*cang[2]]])
    elif rtype == "xyz":
        return np.array([[cang[1]*cang[2], sang[0]*sang[1]*cang[2] + cang[0]*sang[2], -cang[0]*sang[1]*cang[2] + sang[0]*sang[2]],
                         [-cang[1]*sang[2], -sang[0]*sang[1]*sang[2] + cang[0]
                             * cang[2], cang[0]*sang[1]*sang[2] + sang[0]*cang[2]],
                         [sang[1], -sang[0]*cang[1], cang[0]*cang[1]]])
    elif rtype == "xyx":
        return np.array([[cang[1], sang[0]*sang[1], -cang[0]*sang[1]],
                         [sang[1]*sang[2], -sang[0]*cang[1]*sang[2] + cang[0]
                             * cang[2], cang[0]*cang[1]*sang[2] + sang[0]*cang[2]],
                         [sang[1]*cang[2], -sang[0]*cang[2]*cang[1] - cang[0]*sang[2], cang[0]*cang[2]*cang[1] - sang[0]*sang[2]]])
    elif rtype == "xzy":
        return np.array([[cang[2]*cang[1], cang[0]*cang[2]*sang[1] + sang[0]*sang[2], sang[0]*cang[2]*sang[1] - cang[0]*sang[2]],
                         [-sang[1], cang[0]*cang[1], sang[0]*cang[1]],
                         [sang[2]*cang[1], cang[0]*sang[1]*sang[2] - sang[0]*cang[2], sang[0]*sang[1]*sang[2] + cang[0]*cang[2]]])
    elif rtype == "xzx":
        return np.array([[cang[1], cang[0]*sang[1], sang[0]*sang[1]],
                         [-sang[1]*cang[2], cang[0]*cang[2]*cang[1] - sang[0]
                             * sang[2], sang[0]*cang[2]*cang[1] + cang[0]*sang[2]],
                         [sang[1]*sang[2], -cang[0]*cang[1]*sang[2] - sang[0]*cang[2], -sang[0]*cang[1]*sang[2] + cang[0]*cang[2]]])
    else:
        raise ValueError('Invalid rotation order')


@nb.njit(cache=True)
def angles2quats(angles, rtype='zyx', unit='rad'):
    dcm = angles2dcm(angles, rtype=rtype, unit=unit)
    return dcm2quats(dcm)


@nb.njit(cache=True)
def quats2angles(q):
    phi = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]),
                     (q[0]**2+q[3]**2-q[1]**2-q[2]**2))
    theta = np.arcsin(2*(q[0]*q[2]-q[1]*q[3]))
    psi = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),
                     (q[0]**2+q[1]**2-q[2]**2-q[3]**2))

    return np.array([phi, theta, psi])


@nb.njit(cache=True)
def quats2dcm(q):
    DCM = np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*q[1]*q[2]+2*q[0]*q[3], 2*q[1]*q[3]-2*q[0]*q[2]],
                    [2*q[1]*q[2]-2*q[0]*q[3], q[0]**2-q[1]**2 +
                        q[2]**2-q[3]**2, 2*q[2]*q[3]+2*q[0]*q[1]],
                    [2*q[1]*q[3]+2*q[0]*q[2], 2*q[2]*q[3]-2*q[0]*q[1], q[0]**2-q[1]**2-q[2]**2+q[3]**2]])
    return DCM


@nb.njit(cache=True)
def omega2qdot(omega, quat, k=1.0):
    p = omega[0]
    q = omega[1]
    r = omega[2]

    e = k * (1-(quat[0]*quat[0]+quat[1]*quat[1] +
                quat[2]*quat[2]+quat[3]*quat[3]))

    qdot = 0.5*np.array([[e*quat[0] - p*quat[1] - q*quat[2] - r*quat[3]],
                         [p*quat[0] + e*quat[1] + r*quat[2] - q*quat[3]],
                         [q*quat[0] - r*quat[1] + e*quat[2] + p*quat[3]],
                         [r*quat[0] + q*quat[1] - p*quat[2] + e*quat[3]]])

    return qdot.flatten()


@nb.njit(cache=True)
def qdot2omega(qdot):
    raise NotImplementedError('qdot2omega not implemented yet')

@nb.njit(cache=True)
def rot_x(angle):
    t_x = np.array([[1., 0., 0.],
                    [0., np.cos(angle), np.sin(angle)],
                    [0., -np.sin(angle), np.cos(angle)]])

    return t_x

if __name__ == "__main__":
    print("lol")
