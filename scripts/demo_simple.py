from odessa.simulation import Simulation
from odessa.helpers.frames import ecef_to_wgs84
from pathlib import Path

import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = "config_6dof.json"

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
    lla = np.array([ecef_to_wgs84([sol.y[0][i], sol.y[1][i], sol.y[2][i]])
                    for i in range(sol.y[0].shape[0])])
    ax.plot(lla[:, 0], lla[:, 1], lla[:, 2])

plt.show()
