from odessa.simulation import Simulation
from odessa.helpers.frames import wgs84_to_ecef, ecef_to_wgs84

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os
import warnings
warnings.simplefilter('ignore')

filename = "config_6dof_aero.json"

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, filename)
config = Path(filename).read_text()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


np.set_printoptions(threshold=np.inf)
downrange = []
x = np.linspace(0, 360, 25)
y_plot = []
for scale in [0, 1]:
    for heading in x:
        sim = Simulation.fromJSON(config)
        sim.dt = 1
        # del sim.phases[0].modules["AeroLinear"]
        # del sim.phases[1].modules["AeroLinear"]
        # del sim.phases[2].modules["AeroLinear"]

        sim.phases[0].modules["AeroLinear"].scale = np.ones(6, dtype=np.float64)*scale
        sim.phases[1].modules["AeroLinear"].scale = np.ones(6, dtype=np.float64)*scale
        sim.phases[2].modules["AeroLinear"].scale = np.ones(6, dtype=np.float64)*scale

        sim.phases[0].modules["TowerSpherical"].heading = heading
        # sim.phases[1].modules["Thrust"].thrust = 0
        sim.phases[0].modules["TowerSpherical"].elevation = 80
        # sim.phases.pop(2)
        # sim.phases.pop(1)

        sols = sim.run()

        for m, sol in enumerate(sols):
            y = []
            for n, t in enumerate(sol.t):
                pos = sol.y[:3, n]
                y_t = ecef_to_wgs84(pos)
                y.append(y_t)
            y = np.array(y)
            ax.plot(y[:, 0], y[:, 1], y[:, 2], c=f"C{scale}")
            if m == 0:
                # print(sol.t[-1])
                # y_plot.append(np.linalg.norm(y[-1, 0:2]))
                init_pos = y[0, 0:3]

        final_pos = y[-1, 0:3]
        downrange.append(np.linalg.norm(
            wgs84_to_ecef(final_pos) - wgs84_to_ecef(init_pos)))

# print(downrange)
# print(sum(downrange)/len(downrange))
    # break

# vec = np.array(vec)
# ax.quiver(y[:,0], y[:,1], y[:,2], vec[:,0], vec[:,1], vec[:,2], length=0.0005, normalize=True)
# ax.plot(vec[:,0], vec[:,1], vec[:,2])
ax.set_xlabel('Latitude [rad]')
ax.set_ylabel('Longitude [rad]')
ax.set_zlabel('Altitude [m]')
# plt.plot(x ,y_plot)
# print(downrange)
plt.show()
