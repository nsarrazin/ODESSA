import numba as nb
import numpy as np
from ...helpers.transformations import TEC
from ...helpers.frames import ecef_to_wgs84


class apogeeNED:
    """
    Creates an event function that triggers when apogee is reached in NED frame. 
    If the velocity downward changes sign then event is triggered.
    """

    def __init__(self):
        self.terminal = True

    @property
    def event(self):
        @nb.njit
        def event(t, y):
            delta, tau, r = ecef_to_wgs84(y[0:3])
            DCMec = TEC(tau, delta)

            vel_c = y[3:6]
            vel_e = DCMec @ vel_c
            
            return vel_e[2]

        event.terminal = self.terminal
        return event
