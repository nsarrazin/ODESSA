from odessa.simulation import Simulation
from odessa.helpers.frames import ecef_to_wgs84
from pathlib import Path

import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = "data_s4/stratos4.json"

dirname = os.path.dirname(__file__)

filename = os.path.join(dirname, filename)
config = Path(filename).read_text()

# running the thing
sim = Simulation.fromJSON(config)
sols = sim.run()

fig = plt.figure()
history = sim.history
plt.plot(history["t"], history["alpha"], label="alpha")
plt.plot(history["t"], history["beta"], label="beta")
plt.legend()
plt.show()
