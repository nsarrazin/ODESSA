import numba as nb
import numpy as np
from ...helpers.frames import ecef_to_wgs84


class groundFlat:
    """
    Creates an event function that triggers when the ground is reached
    in a flat set of coordinates
    """

    def __init__(self):
        self.terminal = True

    @property
    def event(self):
        @nb.njit
        def event(t, y):
            return y[2]

        event.terminal = self.terminal
        return event


class groundLLA:
    """
    Creates an event function that triggers when the ground is reached
    in the WGS-84 set of coordinates.
    """

    def __init__(self):
        self.terminal = True

    @property
    def event(self):
        @nb.njit
        def event(t, y):
            lla = ecef_to_wgs84(np.array([y[0], y[1], y[2]]))
            return lla[2]+0.0000001

        event.terminal = self.terminal
        return event
