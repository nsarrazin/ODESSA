""""
NASA test case atmos 01: Dragless sphere.
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


def test_case01(plotting=False):
    FT_TO_M = 0.3048
    SLUGFT2_TO_KGM2 = 1.35581795

    # Sphere parameters
    Cd = 0
    S = 0.1963495 * FT_TO_M**2  # m^2
    m = 14.5939                 # kg (= 1 slug)
    Ixx = 3.6*SLUGFT2_TO_KGM2   # kg m^2 == Iyy == Izz
    I = np.array([[Ixx, 0.0, 0.0], [0.0, Ixx, 0.0], [0.0, 0.0, Ixx]])

    # read data from csv files to compare results and set up I.C.s
    nasa_sims_N = 6

    dirname = os.path.dirname(__file__)

    nasa_data = {}
    for i in range(nasa_sims_N):
        if i == 2:
            # third file is different and doesn't contain all data
            continue

        data_path_csv = "Atmos_01_SphereNoDrag/Atmos_01_sim_0{}.csv".format(i+1)
        filename_csv = os.path.join(dirname, data_path_csv)
        data = pd.read_csv(filename_csv, sep=',')

        # read all the data from the pandas array
        nasa_t = data["time"]

        nasa_lon = data["longitude_deg"]
        nasa_lat = data["latitude_deg"]
        nasa_alt = data["altitudeMsl_ft"] * FT_TO_M

        nasa_vx = data["feVelocity_ft_s_X"] * FT_TO_M
        nasa_vy = data["feVelocity_ft_s_Y"] * FT_TO_M
        nasa_vz = data["feVelocity_ft_s_Z"] * FT_TO_M

        nasa_phi = data["eulerAngle_deg_Roll"]
        nasa_theta = data["eulerAngle_deg_Pitch"]
        nasa_psi = data["eulerAngle_deg_Yaw"]

        nasa_p = data["bodyAngularRateWrtEi_deg_s_Roll"]
        nasa_q = data["bodyAngularRateWrtEi_deg_s_Pitch"]
        nasa_r = data["bodyAngularRateWrtEi_deg_s_Yaw"]

        nasa_lla = np.vstack((nasa_lat, nasa_lon, nasa_alt))
        nasa_vel = np.vstack((nasa_vx, nasa_vy, nasa_vz))
        nasa_omega = np.vstack((nasa_p, nasa_q, nasa_r))

        nasa_data[i+1] = {"nasa_t":nasa_t, "nasa_lon":nasa_lon, "nasa_lat":nasa_lat, "nasa_alt":nasa_alt, "nasa_vx":nasa_vx,
                        "nasa_vy":nasa_vy, "nasa_vz":nasa_vz, "nasa_phi":nasa_phi, "nasa_theta":nasa_theta,
                        "nasa_psi":nasa_psi, "nasa_p":nasa_p, "nasa_q":nasa_q, "nasa_r":nasa_r}

        if i == 0 or i == 4 or i == 5:
            # only these have the right ECEF positions
            nasa_x = data["gePosition_ft_X"] * FT_TO_M
            nasa_y = data["gePosition_ft_Y"] * FT_TO_M
            nasa_z = data["gePosition_ft_Z"] * FT_TO_M

            nasa_data[i+1]["nasa_x"] = nasa_x
            nasa_data[i+1]["nasa_y"] = nasa_y
            nasa_data[i+1]["nasa_z"] = nasa_z

    # initial condition
    ic_lla = np.array([np.deg2rad(nasa_lat[0]), np.deg2rad(nasa_lon[0]), nasa_alt[0]])
    ic_vel = np.array([nasa_vx[0], nasa_vy[0], nasa_vz[0]])
    ic_omega = np.deg2rad(np.array([-0.004178073, 0.0, 0.0]))

    # load the json for atmos case 01
    filename = "config_atmos_01.json"

    filename = os.path.join(dirname, filename)
    config = Path(filename).read_text()

    # create the simulation object
    sim = Simulation.fromJSON(config)

    # modifying the values based on initial conditions
    sim.phases[0].modules["ConstantMassInertia"].mass = m
    sim.phases[0].modules["ConstantMassInertia"].inertia = I

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
        
    if plotting:
        fig, ax = plt.subplots(3, 1)
        fig.suptitle("Longitude, Latitude and Altitude")

        fig2, ax2 = plt.subplots(3, 1)
        fig2.suptitle("Roll, Pitch and Yaw Rates")

        fig3, ax3 = plt.subplots(3, 1)
        fig3.suptitle("Roll, Pitch and Yaw Angles")

        fig4, ax4 = plt.subplots(3, 1)
        fig4.suptitle("XYZ position ECEF")

        ax[0].plot(sol.t, y[:, 2], marker="o", label="ODESSA")
        ax[1].plot(sol.t, np.rad2deg(y[:, 1]), marker="o", label="ODESSA")
        ax[2].plot(sol.t, np.rad2deg(y[:, 0]), marker="o", label="ODESSA")

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
        for i in range(nasa_sims_N):
            if i == 2:
                continue

            label_name = "NASA Sim 0{}".format(i+1)

            ax[0].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_alt"], marker="x", label=label_name)
            ax[1].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_lon"], marker="x", label=label_name)
            ax[2].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_lat"], marker="x", label=label_name)

            ax2[0].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_p"], marker="x", label=label_name)
            ax2[1].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_q"], marker="x", label=label_name)
            ax2[2].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_r"], marker="x", label=label_name)

            ax3[0].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_phi"], marker="x", label=label_name)
            ax3[1].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_theta"], marker="x", label=label_name)
            ax3[2].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_psi"], marker="x", label=label_name)

            if i == 0 or i == 4 or i == 5:
                # only these have the right ECEF positions
                ax4[0].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_x"], marker="x", label=label_name)
                ax4[1].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_y"], marker="x", label=label_name)
                ax4[2].plot(nasa_data[i+1]["nasa_t"], nasa_data[i+1]["nasa_z"], marker="x", label=label_name)

        # add labels to the plot
        ax[0].set(ylabel="Altitude [m]", xlabel="Time [s]")
        ax[0].legend(numpoints=1)
        ax[1].set(ylabel="Longitude [deg]", xlabel="Time [s]")
        ax[1].legend(numpoints=1)
        ax[2].set(ylabel="Latitude [deg]", xlabel="Time [s]")
        ax[2].legend(numpoints=1)

        ax2[0].set(ylabel="p (roll rate) [deg/s]", xlabel="Time [s]",
                title="Note: the initial condition is correctly shown for ODESSA, but the not for NASA")
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
    nasa_f = np.array([np.array(nasa_data[6]["nasa_lat"])[-1], np.array(nasa_data[6]["nasa_lon"])[-1],
                    np.array(nasa_data[6]["nasa_alt"])[-1]])

    errs = np.sqrt((sim_f[0] - nasa_f[0])**2), np.sqrt((sim_f[1] - nasa_f[1])**2), np.sqrt((sim_f[2] - nasa_f[2])**2)

    assert np.amax(errs) < 1
    # print(errs)
    # print("Test passed: {} ({})".format(np.amax(errs) < 1, np.amax(errs)))
