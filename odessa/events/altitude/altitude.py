import numba as nb
import numpy as np
from ...helpers.transformations import TEC
from ...helpers.frames import ecef_to_wgs84


class altitude:
    """
    Creates an event function that triggers when altitude reaches beyond passed altitude.
    If the altitude difference becomes negative then event is triggered.
    """

    def __init__(self):
        self.terminal = True
        self.altitude = 0.
    
    @property
    def event(self):
        @nb.njit
        def event(t, y, altitude):
            delta, tau, r = ecef_to_wgs84(y[0:3])

            return r - altitude

        def event_func(t, y): return event(t, y, self.altitude)

        event_func.terminal = self.terminal
        return event_func
