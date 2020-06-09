from py_rts.modules.core import Core
from py_rts.modules.generator import gen_rhs

from odessa.simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pathlib import Path
import os


def run_rts(params):
    # print(i)
    core = Core()
    core.lla = params["lla"]
    # core.v_inertial = core.x_inertial/np.linalg.norm(core.x_inertial) * v_coeff + v_inertial
    core.v_inertial = v_inertial
    params['w_e'] = core.w_e  # add w_e to pdict
    rhs = gen_rhs(params)

    core.tf = 1200

    core.run(rhs)
    core.post_process()

    history = {k: v for k, v in zip(
        core.state_header, np.array(core.state_history))}

    return [history["x_inertial(0)"], history["x_inertial(1)"], history["x_inertial(2)"]]


def run_3dof(params):
    # print(i)
    # get the filepath to the config file and open it

    sim.phases[0].modules["Mass"].mass = params["mass"]
    sim.phases[0].modules["Aero"].wind_alts = params["alts"]
    sim.phases[0].modules["Aero"].wind_headings = params["wind_headings"]
    sim.phases[0].modules["Aero"].wind_speeds = params["wind_speeds"]

    # set initial state (position, velocity)
    sim.Core.lla = params["lla"]
    sim.Core.vel = v_inertial
    sim.Core.t = 0
    sim.phases[0].modules["Atmos"].init_history()
    # run this bad boy
    sols = sim.run()
    return [sols[-1].y[0], sols[-1].y[1], sols[-1].y[2]]


if __name__ == "__main__":
    distances = []
    speedups = []

    N = 100

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # SETUP FOR ODESSA
    # Explanation : ODESSA uses jitclass a lot. jitclasses do not support ahead of time compiling (yet)
    # Therefore if at all possible, try to reuse the Simulation class, resetting state variables
    # and updating module values between runs if needed
    #
    # Here for example we create the simulation object and we're going to reset Core.vel and Core.lla between runs,
    # but the object will stay the same, so it doesn't have to be compiled again
    #
    # tldr: the less modules you initialize the faster the code so try to reuse objects
    #        a 15x speedup of the code was observed when simulation object is reused here

    dirname = os.path.dirname(__file__)

    # as of right now, wind doesn't work so it'll have to be fixed
    filename = os.path.join(dirname, 'config_benchmark.json')
    config = Path(filename).read_text()

    # create the full simulation from a single json !?!?!
    sim = Simulation.fromJSON(config)

    for i in range(N):
        print(i)
        lat = np.radians(np.random.rand()*180-90)
        lon = np.radians(np.random.rand()*360-180)
        alt = np.random.rand()*18000+2000

        lla = np.array([lat, lon, alt])
        v_inertial = np.array(
            [np.random.rand()*200, np.random.rand()*200, np.random.rand()*200], dtype=np.float64)

        params = {}
        params["alts"] = np.array([0., 999., 1e3, 2e3, 2e3+1, 1e6])
        params["wind_headings"] = np.zeros(6, dtype=np.float64)
        params["wind_speeds"] = np.zeros(6, dtype=np.float64) * 30

        params["lla"] = lla
        params["machs"] = np.array([0., 1., 999])
        params["drags"] = np.array([1., 1., 1.])
        params["mass"] = np.random.rand()*490 + 10.

        sol_rts = run_rts(params)
        sol_3dof = run_3dof(params)

        t0 = time.time()
        for fromJSON in range(10):
            run_rts(params)
        t1 = time.time()
        t_sol_rts = t1-t0

        t0 = time.time()
        for j in range(10):
            run_3dof(params)
        t1 = time.time()
        t_sol_3dof = t1-t0

        MSE_rts_3dof = np.sqrt((sol_rts[0][-1] - sol_3dof[0][-1])**2 +
                               (sol_rts[1][-1] - sol_3dof[1][-1])**2 +
                               (sol_rts[2][-1] - sol_3dof[2][-1])**2)

        distances.append(MSE_rts_3dof)
        speedups.append(t_sol_rts/t_sol_3dof)

        threshold = 100
        if MSE_rts_3dof > threshold:
            print("WARNING - Error of {:.2f}m detected.".format(MSE_rts_3dof))
            ax.plot(sol_3dof[0], sol_3dof[1],  sol_3dof[2], C="C0")
            ax.plot(sol_rts[0], sol_rts[1],  sol_rts[2], c="C1")
            plt.show()

            # break
    # worst cases :
    print(f"Number of test cases ran    : {i+1}")
    print(f"Max error vs pyrts          :  {np.amax(distances):.3f}m")
    print(f"Max slowdown vs pyrts       : x{np.amin(speedups):.3f}")
