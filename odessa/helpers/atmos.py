import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def atmo_isa(alt):
    """This function computes the ISA atmosphere as a function of altitude.

    Arguments:
        alt {[float64]} -- The altitude from the ground.

    Returns:
        [float64] -- [The temperature in K]
        [float64] -- [The pressure in Pa]
        [float64] -- [The density in kg/m³]
        [float64] -- [The speed of sound in m/s]

    """
    layers = np.array([-610,      11000, 20000,  32000,
                       47000,   51000,   71000, 84852])
    lapse_rates = np.array(
        [-0.0065,   0, 0.001, 0.0028,     0, -0.0028,  -0.002,     0])
    base_T = np.array([292.15, 216.685, 216.685, 228.685,
                       270.685, 270.685, 214.685, 186.981])
    base_P = np.array([108900, 22643.07, 5478.8, 868.89,
                       111.05, 67.03, 3.96, 0.374])
    R = 287.05287
    g0_R = 0.03416321878

    if alt > 84850:
        alt = 84850
    if alt < 0:
        alt = 0

    for i in range(8):
        if layers[i] > alt:
            layer_num = i - 1
            break

    dh = alt - layers[layer_num]

    temp = base_T[layer_num] + lapse_rates[layer_num] * dh

    if layer_num == 1 or layer_num == 4:
        pressure = base_P[layer_num] * np.exp(-g0_R * dh/temp)
    else:
        pressure = base_P[layer_num] * \
            np.power(temp / base_T[layer_num], -g0_R/lapse_rates[layer_num])

    density = pressure / (R*temp)
    speed_sound = np.sqrt(R*1.4*temp)

    return temp, pressure, density, speed_sound
