""""
NASA test case orbit 06A: Orbital sphere with constant drag.
"""

from odessa.simulation import Simulation
from odessa.helpers.frames import ecef_to_wgs84
from odessa.helpers.rotation import quats2angles

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import warnings
warnings.simplefilter('ignore')


FT_TO_M = 0.3048
SLUGFT2_TO_KGM2 = 1.35581795
SLUG_TO_KG = 14.5939

# orbital sphere parameter
Cd = 0.02
radius = 1/np.sqrt(np.pi)  # m
S = 1.0  # m^2
m = 1.0  # kg
Ixx = 2/(5*np.pi)  # kg m^2
Iyy = 2/(5*np.pi)  # kg m^2
Izz = 2/(5*np.pi)  # kg m^2
I = np.array([[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])

# read data from csv files to compare results and set up I.C.s
nasa_sims_N = ["A", "B", "C", "D"]

dirname = os.path.dirname(__file__)

nasa_data = {}
for i in nasa_sims_N:

    if i == "A":
        # Simulator A has no data for this case
        continue

    data_path_csv = "Orbit_06A_SphereFixedDrag/Orbit_06a_sim_{}.csv".format(i)
    filename_csv = os.path.join(dirname, data_path_csv)
    data = pd.read_csv(filename_csv, sep=',')

    # read all the data from the pandas array
    nasa_t = data["time"]

    nasa_phi = data["eulerAngle_rad_Roll"]
    nasa_theta = data["eulerAngle_rad_Pitch"]
    nasa_psi = data["eulerAngle_rad_Yaw"]

    nasa_p = data["bodyAngularRateWrtEi_rad_s_Roll"]
    nasa_q = data["bodyAngularRateWrtEi_rad_s_Pitch"]
    nasa_r = data["bodyAngularRateWrtEi_rad_s_Yaw"]

    nasa_omega = np.vstack((nasa_p, nasa_q, nasa_r))

    nasa_data[i] = {"nasa_t":nasa_t, "nasa_phi":nasa_phi, "nasa_theta":nasa_theta, "nasa_psi":nasa_psi, "nasa_p":nasa_p,
                    "nasa_q":nasa_q, "nasa_r":nasa_r}

    nasa_x = data["gePosition_m_X"]
    nasa_y = data["gePosition_m_Y"]
    nasa_z = data["gePosition_m_Z"]

    nasa_data[i]["nasa_x"] = nasa_x
    nasa_data[i]["nasa_y"] = nasa_y
    nasa_data[i]["nasa_z"] = nasa_z

# initial condition
ic_lla = np.array([0.0, 0.0, nasa_alt[0]])
ic_vel = np.array([nasa_vx[0], nasa_vy[0], nasa_vz[0]])
ic_omega = np.deg2rad(np.array([9.995821927, 20.0, 30.0]))

# load the json for atmos case 01
filename = "config_orbit_06A.json"

filename = os.path.join(dirname, filename)
config = Path(filename).read_text()

# create the simulation object
sim = Simulation.fromJSON(config)

# modifying the values based on initial conditions
sim.phases[0].modules["ConstantMassInertia"].mass = m
sim.phases[0].modules["ConstantMassInertia"].inertia = I

sim.phases[0].modules["AeroConstant"].bref = radius
sim.phases[0].modules["AeroConstant"].cref = radius
sim.phases[0].modules["AeroConstant"].sref = S
sim.phases[0].modules["AeroConstant"].c_x_0 = Cd

sim.Core.vel = ic_vel
sim.Core.lla = ic_lla
sim.Core.omega = ic_omega

# we get rid of the ground trigger and make sure the integrator just stops at 30s
sim.phases[0].event = lambda t, y: 1

sim.tf = 30
sim.dt = 0.1

# actual simulation
sols = sim.run()


# process results from simulation
fig2, ax2 = plt.subplots(3, 1)
fig2.suptitle("Roll, Pitch and Yaw Rates")

fig3, ax3 = plt.subplots(3, 1)
fig3.suptitle("Roll, Pitch and Yaw Angles")

fig4, ax4 = plt.subplots(3, 1)
fig4.suptitle("XYZ position ECEF")

for m, sol in enumerate(sols):
    y = []
    y_ecef = []
    vel = []
    angles = []
    rates = []

    for n, t in enumerate(sol.t):
        pos = sol.y[:3, n]
        y_t = ecef_to_wgs84(pos)
        y.append(y_t)
        y_ecef.append(pos)

        vel.append(sol.y[3:6, n])

        angles.append(quats2angles(sol.y[6:10, n]))

        rates.append(sol.y[10:13, n])

    y = np.array(y)
    y_ecef = np.array(y_ecef)
    vel = np.array(vel)
    angles = np.array(angles)
    rates = np.rad2deg(np.array(rates))

    ax2[0].plot(sol.t, rates[:, 0], marker="o", label="ODESSA")
    ax2[1].plot(sol.t, rates[:, 1], marker="o", label="ODESSA")
    ax2[2].plot(sol.t, rates[:, 2], marker="o", label="ODESSA")

    ax3[0].plot(sol.t, np.rad2deg(angles[:, 0]), marker="o", label="ODESSA")
    ax3[1].plot(sol.t, np.rad2deg(angles[:, 1]), marker="o", label="ODESSA")
    ax3[2].plot(sol.t, np.rad2deg(angles[:, 2]), marker="o", label="ODESSA")

    ax4[0].plot(sol.t, y_ecef[:, 0], marker="o", label="ODESSA")
    ax4[1].plot(sol.t, y_ecef[:, 1], marker="o", label="ODESSA")
    ax4[2].plot(sol.t, y_ecef[:, 2], marker="o", label="ODESSA")


# plot data from NASA test cases
for i in nasa_sims_N:
    if i == "A":
        continue

    label_name = "NASA Sim 0{}".format(i)

    ax2[0].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_p"], marker="x", label=label_name)
    ax2[1].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_q"], marker="x", label=label_name)
    ax2[2].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_r"], marker="x", label=label_name)

    ax3[0].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_phi"], marker="x", label=label_name)
    ax3[1].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_theta"], marker="x", label=label_name)
    ax3[2].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_psi"], marker="x", label=label_name)

    ax4[0].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_x"], marker="x", label=label_name)
    ax4[1].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_y"], marker="x", label=label_name)
    ax4[2].plot(nasa_data[i]["nasa_t"], nasa_data[i]["nasa_z"], marker="x", label=label_name)

# add labels to the plot
ax2[0].set(ylabel="p (roll rate) [deg/s]", xlabel="Time [s]")
ax2[0].legend(numpoints=1)
ax2[1].set(ylabel="q (pitch rate) [deg/s]", xlabel="Time [s]")
ax2[1].legend(numpoints=1)
ax2[2].set(ylabel="r (yaw rate) [deg/s]", xlabel="Time [s]")
ax2[2].legend(numpoints=1)

ax3[0].set(ylabel=r"$\phi$ (roll angle) [deg]", xlabel="Time [s]")
ax3[0].legend(numpoints=1)
ax3[1].set(ylabel=r"$\theta$ (pitch angle) [deg]", xlabel="Time [s]")
ax3[1].legend(numpoints=1)
ax3[2].set(ylabel=r"$\psi$ (yaw angle) [deg]", xlabel="Time [s]")
ax3[2].legend(numpoints=1)

ax4[0].set(ylabel="X [m]", xlabel="Time [s]")
ax4[0].legend(numpoints=1)
ax4[1].set(ylabel="Y [m]", xlabel="Time [s]")
ax4[1].legend(numpoints=1)
ax4[2].set(ylabel="Z [m]", xlabel="Time [s]")
ax4[2].legend(numpoints=1)

plt.show()


sim_f = np.array([np.rad2deg(y[:,0][-1]), np.rad2deg(y[:,1][-1]), y[:,2][-1]])
nasa_f = np.array([np.array(nasa_data["A"]["nasa_lat"])[-1], np.array(nasa_data["A"]["nasa_lon"])[-1],
                   np.array(nasa_data["A"]["nasa_alt"])[-1]])
errs = np.sqrt((sim_f[0] - nasa_f[0])**2), np.sqrt((sim_f[1] - nasa_f[1])**2), np.sqrt((sim_f[2] - nasa_f[2])**2)

sim_f2 = np.array([np.rad2deg(angles[:,0][-1]), np.rad2deg(angles[:,1][-1]), np.rad2deg(angles[:,2][-1])])
nasa_f2 = np.array([np.array(nasa_data["A"]["nasa_phi"])[-1], np.array(nasa_data["A"]["nasa_theta"])[-1],
                   np.array(nasa_data["A"]["nasa_psi"])[-1]])
errs2 = np.sqrt((sim_f2[0] - nasa_f2[0])**2), np.sqrt((sim_f2[1] - nasa_f2[1])**2), np.sqrt((sim_f2[2] - nasa_f2[2])**2)

print(errs)
print("EoM passed: {} ({})".format(np.amax(errs) < 1, np.amax(errs)))
print(errs2)
print("Angles passed: {} ({})".format(np.amax(errs2) < 1, np.amax(errs2)))
