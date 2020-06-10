from odessa.simulation import Simulation
from odessa.helpers.frames import wgs84_to_ecef, ecef_to_wgs84

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os
import warnings

# This demo script runs a 6DoF trajectory with thrust and drag at multiple launch angles

warnings.simplefilter('ignore')
np.set_printoptions(threshold=np.inf)

filename = "configs/config_6dof_aero.json"
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, filename)
config = Path(filename).read_text()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
downrange = []

x = np.linspace(0, 360, 10)
y_plot = []

for scale in [0, 1]:
    for heading in x:
        sim = Simulation.fromJSON(config)
        sim.dt = 1

        sim.phases[0].modules["AeroLinear"].scale = np.ones(6, dtype=np.float64)*scale
        sim.phases[1].modules["AeroLinear"].scale = np.ones(6, dtype=np.float64)*scale
        sim.phases[2].modules["AeroLinear"].scale = np.ones(6, dtype=np.float64)*scale

        sim.phases[0].modules["TowerSpherical"].heading = heading

        sim.phases[0].modules["TowerSpherical"].elevation = 80

        sols = sim.run()

        # plotting the trajectory
        for m, sol in enumerate(sols):
            y = []
            for n, t in enumerate(sol.t):
                pos = sol.y[:3, n]
                y_t = ecef_to_wgs84(pos)
                y.append(y_t)
            y = np.array(y)
            ax.plot(y[:, 0], y[:, 1], y[:, 2], c=f"C{scale}")
            if m == 0:
                init_pos = y[0, 0:3]

ax.set_xlabel('Latitude [rad]')
ax.set_ylabel('Longitude [rad]')
ax.set_zlabel('Altitude [m]')
plt.show()
