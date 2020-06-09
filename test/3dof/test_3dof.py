from odessa.simulation import Simulation
from odessa.helpers.frames import wgs84_to_ecef, ecef_to_wgs84
from odessa.helpers.transformations import TCE, ecef2spherical
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import os

import warnings
warnings.simplefilter('ignore')

def run_case(CASE, plotting=False):
    """Runs the test case from NASA based on the case number.

    Args:
        CASE ([int]): Case number to be tested, can be 6,7,8,9 or 10.
        plotting (bool, optional): Whether or not the resulting trajectory should be plotted.
        Defaults to False.

    Raises:
        ValueError: Case number doesn't exist.

    Returns:
        err_x, err_y, err_z : The error in all dimensions.
    """
    FT_TO_M = 0.3048

    # used for all cases
    Cd = 0.1
    S = 0.0182414654525
    m = 14.5939

    # used for case 7
    wind_v = 20*FT_TO_M
    wind_heading = np.radians(270)
    # used for case 8
    wind_v_shear1 = 70*FT_TO_M
    wind_v_shear2 = -20*FT_TO_M

    if CASE not in [6, 7, 8, 9, 10]:
        raise ValueError("CASE should be 6, 7, 8, 9 or 10")

    # read the NASA datafile
    if CASE >= 10:
        data_path = f"configs/Atmos_{CASE}_sim_01.csv"
    else:
        data_path = f"configs/Atmos_0{CASE}_sim_01.csv"

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, data_path)

    data = pd.read_csv(filename, sep=',')

    # read all the data from the pandas array
    lon = np.radians(data["longitude_deg"])
    lat = np.radians(data["latitude_deg"])
    alt = data["altitudeMsl_ft"] * FT_TO_M

    vx = data["feVelocity_ft_s_X"] * FT_TO_M
    vy = data["feVelocity_ft_s_Y"] * FT_TO_M
    vz = data["feVelocity_ft_s_Z"] * FT_TO_M

    nasa_lla = np.vstack((lat, lon, alt))
    nasa_vel = np.vstack((vx, vy, vz))

    # load the json for ODESSA
    filename = "configs/config_rts.json"  # path to the ODESSA json

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filename)
    config = Path(filename).read_text()

    # create the simulation object
    sim = Simulation.fromJSON(config)

    # modifying the values based on NASA data
    sim.phases[0].modules["ConstantMass"].mass = m
    # constant drag so interpolator is constant from 0 to 1e6
    sim.phases[0].modules["AeroRTS"].machs = np.array(
        [0, 1e6], dtype=np.float64)
    sim.phases[0].modules["AeroRTS"].drags = np.ones(
        2, dtype=np.float64)*Cd*S  # drag is equal to Cd*S

    # we get rid of the ground trigger and make sure the integrator just stops at 30s
    sim.phases[0].event = lambda t, y: 1

    sim.tf = 30
    sim.dt = 0.1

    # for specific cases we need some extra parameters, instead of making a lot of JSONs, I just modify the parameters on the fly

    if CASE == 7:  # constant wind
        vn = np.cos(wind_heading) * wind_v
        ve = np.sin(wind_heading) * wind_v


        sim.phases[0].modules["AeroRTS"].wind_alts = np.array(
            [0, 1e6], dtype=np.float64)
        sim.phases[0].modules["AeroRTS"].wind_speeds = np.array([[vn, ve], [vn, ve]])  # constant wind speed

    elif CASE == 8:  # variable wind
        vn_2 = np.cos(wind_heading) * wind_v_shear2
        ve_2 = np.sin(wind_heading) * wind_v_shear2

        vn_1 = np.cos(wind_heading) * wind_v_shear1
        ve_1 = np.sin(wind_heading) * wind_v_shear1

        sim.phases[0].modules["AeroRTS"].wind_alts = np.array(
            [0, 30000*FT_TO_M], dtype=np.float64)
        sim.phases[0].modules["AeroRTS"].wind_speeds = np.array(
            [[vn_2, ve_2], [vn_1, ve_1]], dtype=np.float64)

    elif CASE == 9:  # cannonball on a ballistic trajectory west
        pos = wgs84_to_ecef(nasa_lla[:, 0])  # gives us initial ECEF position
        r, delta, tau = ecef2spherical(pos)
        DCMce = TCE(tau, delta)  # DCM to go from NED to ECEF

        sim.Core.vel = DCMce @ np.array(
            [0, 1000*FT_TO_M, -1000*FT_TO_M], dtype=np.float64)

    elif CASE == 10:  # cannonball on a ballistic trajectory north
        pos = wgs84_to_ecef(nasa_lla[:, 0])  # gives us initial ECEF position
        r, delta, tau = ecef2spherical(pos)
        DCMce = TCE(tau, delta)  # DCM to go from NED to ECEF

        # we needed the DCM bc velocity is expressed in ECEF
        sim.Core.vel = DCMce @ np.array(
            [1000*FT_TO_M, 0, -1000*FT_TO_M], dtype=np.float64)

    sim.Core.lla = nasa_lla[:, 0]  # set the position in the solver
    sols = sim.run()  # actually run this bad boy

    sim_f = sols[0].y[:3, -1]  # get the final ECEF position in the simulation
    # get the final ECEF position in the NASA files
    nasa_f = wgs84_to_ecef(nasa_lla[:, -1])

    err_x, err_y, err_z = np.sqrt((sim_f[0] - nasa_f[0])**2), np.sqrt(
        (sim_f[1] - nasa_f[1])**2), np.sqrt((sim_f[2] - nasa_f[2])**2)

    if plotting:
        for m, sol in enumerate(sols):
            y = []
            for n, t in enumerate(sol.t):
                pos = sol.y[:3, n]
                y_t = ecef_to_wgs84(pos)
                y.append(y_t)
            y = np.array(y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(y[:, 0], y[:, 1], y[:, 2], color="C0", label="ODESSA")
        ax.plot(nasa_lla[0, :], nasa_lla[1, :],
                nasa_lla[2, :], color="C1", label="NASA")
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.legend()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(sol.t[:], y[:, 2], color="C0", label="ODESSA")
        ax.plot(data["time"], nasa_lla[2, :], color="C1", label="NASA")

        ax.set_xlabel('t [s]')
        ax.set_ylabel('h [m]')

        plt.legend()
        plt.show()
    return err_x, err_y, err_z


# the actual test cases picked up by pytest
def test_case6():
    errs = run_case(6)
    assert np.amax(errs) < 1 


def test_case7():
    errs = run_case(7)
    assert np.amax(errs) < 1


def test_case8():
    errs = run_case(8)
    assert np.amax(errs) < 1


def test_case9():
    errs = run_case(9)
    assert np.amax(errs) < 1


def test_case10():
    errs = run_case(10)
    assert np.amax(errs) < 1

if __name__ == "__main__":
    for i in range(6,11):
        print(np.average(run_case(i, plotting=False)))
