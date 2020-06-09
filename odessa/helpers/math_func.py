""" This module deals with math helper functions numbified
    so they can be loaded in the modules"""
import numpy as np
import numba as nb
from numba import double


@nb.jit(nopython=True, cache=True)
def cross_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


@nb.jit(nopython=True, cache=True)
def bisect(a: np.ndarray, x):
    """Similar to bisect.bisect() or bisect.bisect_right(),
       from the built-in library."""
    M = a.size
    for i in range(M):
        if a[i] > x:
            return i
    return M


@nb.jit(nopython=True, cache=True)
def interp_one(x: np.ndarray, xp: np.ndarray, fp: np.ndarray):
    """Similar to numpy.interp, if x is a single value."""
    i = bisect(xp, x)

    # These edge return values are set with left= and right= in np.interp.
    if i == 0:
        return fp[0]
    elif i == xp.size:
        return fp[-1]

    interp_port = (x - xp[i-1]) / (xp[i] - xp[i-1])

    return fp[i-1] + (interp_port * (fp[i] - fp[i-1]))

@nb.jit(nopython=True, cache=True)
def time_diff(x: np.ndarray, t: np.ndarray, y: np.ndarray):
    """Returns the local derivative between the two nearest points

    Arguments:
        x {np.ndarray} -- [description]
        t {np.ndarray} -- [description]
        y {np.ndarray} -- [description]
    
    Returns:
        [np.ndarray] -- [description]
    """
    i = bisect(t, x)
    
    if i == 0:
        dt = np.abs((y[i+1] - y[i]) / t[i+1] - t[i])
        return dt

    elif i == t.size:
        return 0.
    
    dt = np.abs((y[i] - y[i-1])/(t[i] - t[i-1]))

    return dt


@nb.njit(cache=True)
def n3_grid_interpolator(x_new, data_vector, value_vector):

    length_vector = len(data_vector[:, -1])
    result = 0.

    first_dim_factor = langrangian_bases(data_vector[:, 0], x_new)
    second_dim_factor = langrangian_bases(data_vector[:, 1], x_new)
    third_dim_factor = langrangian_bases(data_vector[:, 2], x_new)

    for i in value_vector[:, 0]:
        for j in value_vector[:, 1]:
            for k in value_vector[:, 2]:
                pass


@nb.njit(cache=True)
def langrangian_bases(dim_vector, x_new):
    ''' Verified by hand '''
    length_dim = len(dim_vector)
    result = 0.

    for i in range(length_dim):
        subresult = 1.
        for value in dim_vector:

            if dim_vector[i] != value:
                subresult *= (x_new - value) / (dim_vector[i] - value)

        result += subresult

    return result

@nb.jit(nopython=True)
def lerp(t, y0, y1):
    '''
    Linear interpolation between two values, given the normalized distance (t) between the two points.
    The function has been checked by setting (0., 1., 0.5), (0.,1., 0.5), (0.5, 1., 0.5)

    :param t: normalized distance between two points
    :param y0: left-bounded value to be interpolated between
    :param y1: right-bounded value to be interpolated between
    :return: interpolated value between y0 and y1
    '''
    return (1. - t) * y0 + t * y1

@nb.jit(nopython=True)
def clip(x, xmin, xmax):
    '''
    Ensures that the passed value is between the values passed in xmin and xmax.

    :param x: value to be checked
    :param xmin: minimum value
    :param xmax: maximum value
    :return: clip: clipped value if x is outside xmin - xrange, else x
    '''
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x

if __name__ == "__main__":
    from scipy.interpolate import RegularGridInterpolator
    x = 2
    t = np.array([0., 1., 3.])
    y = np.array([0., 1., 2.])

    print(time_diff(x, t, y))