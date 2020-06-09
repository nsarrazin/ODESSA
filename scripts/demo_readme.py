from odessa.simulation import Simulation
from odessa.helpers.frames import ecef_to_wgs84

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import os

plt.style.use('ggplot')

fig = plt.figure()

# read the JSON
filepath = 'config_demo.json'

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, filepath)
config = Path(filename).read_text()

# create a Simulation object from JSON
sim = Simulation.fromJSON(config)

# actually run the simulation
sols = sim.run()

for sol in sols:
    lla = np.array([ecef_to_wgs84([sol.y[0][i],sol.y[1][i],sol.y[2][i]]) for i in range(sol.y[0].shape[0])])
    plt.plot(sol.t, lla[:, 2], label='With drag')

# run it again without aerodynamics
sim = Simulation.fromJSON(config)
del sim.phases[0].modules["AeroRTS"]
sols = sim.run()

for sol in sols:
    lla = np.array([ecef_to_wgs84([sol.y[0][i],sol.y[1][i],sol.y[2][i]]) for i in range(sol.y[0].shape[0])])
    plt.plot(sol.t, lla[:, 2], label='Without drag')

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.show()
