import numba as nb
import numpy as np
from ...helpers.frames import wgs84_to_ecef


class distancePointLLA:
    """Creates an event function that triggers on the edge of a sphere
        of a specified radius

    Arguments:
        lla {[float64[:]]} -- [LLA[rad,rad,m] the position of the point at
                              the center of the sphere (usually bottom of
                                                        the launch tower)]
        distance {[float64]} -- [The radius of the sphere, distance at which
                                 the event is triggered]


    if direction > 0, triggers when leaving the bubble of radius dist
    if direction < 0 , triggers when entering the zone
    if direction == 0, both

    """

    def __init__(self):
        self.terminal = True
        self.lla_0 = np.zeros(3)
        self.distance = 0.

    @property
    def event(self):
        @nb.njit
        def distLLA(t, y, lla_0, distance):
            pos_0 = wgs84_to_ecef(lla_0)
            norm = np.linalg.norm(pos_0 - y[0:3])

            return norm - distance

        def event(t, y): return distLLA(t, y, self.lla_0, self.distance)

        event.terminal = self.terminal
        event.direction = 1
        return event
