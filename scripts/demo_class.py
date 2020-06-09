from odessa.simulation import Simulation
from odessa.helpers.frames import ecef_to_wgs84
from odessa.helpers.transformations import TIC
from pathlib import Path

import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def demo(filename, plot='lla'):
    # loading the json
    dirname = os.path.dirname(__file__)

    filename = os.path.join(dirname, filename)
    config = Path(filename).read_text()

    # running the thing
    sim = Simulation.fromJSON(config)
    sols = sim.run()

    # plot plot plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for sol in sols:
        if plot == "lla":
            lla = np.array([ecef_to_wgs84([sol.y[0][i], sol.y[1][i], sol.y[2][i]])
                            for i in range(sol.y[0].shape[0])])
            ax.plot(lla[:, 0], lla[:, 1], lla[:, 2])

        if plot == "eci":
            x, y, z = [], [], []
            for n, t in enumerate(sol.t):
                pos = sol.y[:3, n]
                pos_eci = TIC(t) @ pos
                x.append(pos_eci[0])
                y.append(pos_eci[1])
                z.append(pos_eci[2])

            ax.plot(x, y, z)
    print(sol.t[-1])
    plt.show()
    return sim.JSON


if __name__ == "__main__":
    print("Orbit")
    print(demo('config_orbit.json', plot="eci"))
    print("Multi-phase")
    print(demo('config_phases.json', plot="lla"))
