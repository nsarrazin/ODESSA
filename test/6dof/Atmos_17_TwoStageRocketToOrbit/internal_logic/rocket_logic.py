import numpy as np
from odessa.helpers.math_func import interp_one
from pathlib import Path

# This is a script to get the logic from the NASA test case A17 from dml (xml) files to csv files from which Python
# can consequently use it to make an linear interpolation object. All hard-coded numbers can be found in the NASA
# checkcase appendices, just google NESC 6-DoF check cases.

results_dir_path = Path(__file__).parent / "results"

# CONVERSIONS
FT2M = 0.3048
gravity = 9.8066
LBFTSQ2NM = 1.3558179483314
LBF2N = 4.44822162

# (ROCKET) CONSTANTS
omega_t = 7.2921235169904e-5

inertia_stage1_start = np.array([353250., 33501637.473461, 33501637.473461])
inertia_stage1_end = np.array([150750., 10886636.572139, 10886636.572139])
inertia_stage2_start = np.array([111735., 941063.762626, 941063.762626])
inertia_stage2_end = np.array([21375., 212384.868421, 212384.868421])

ixx_range_stage1, iyy_range_stage1, dummy = inertia_stage1_start - inertia_stage1_end
ixx_range_stage2, iyy_range_stage2, dummy = inertia_stage2_start - inertia_stage2_end

mass = np.array([314000., 134000., 99000., 19000.])

fuel_cap_stage1 = mass[0] - mass[1]
fuel_cap_stage2 = mass[2] - mass[3]

cg_stage1_start = np.array([-16.918700 , 0., 0.])
cg_stage1_end = np.array([-9.421642, 0., 0.])
cg_stage2_start = np.array([-4.797980, 0., 0.])
cg_stage2_end = np.array([-3.947368, 0., 0.])

xcg_range_stage1 = cg_stage1_start - cg_stage1_end
xcg_range_stage2 = cg_stage2_start - cg_stage2_end

mrc_stage1 = np.array([-16.918790, 0., 0.])
mrc_stage2 = np.array([-4.797980, 0., 0.])

lla = np.array([0., 0., 0.])
angles = np.deg2rad(np.array([0., 55.22, 90.]))  # Eb

# Aero
lref = 3.
latref = 3.
sref = 7.

aoa_forces = np.arange(-10., 12., 2.)
aoa_moments = np.array([-20., 0., 20.])

cl = np.array([-1.6, -1.0, -0.73, -0.49, -0.24, 0., 0.24, 0.49, 0.73, 1.0, 1.6])
cd = np.array([0.48, 0.38, 0.31, 0.25, 0.23, 0.21, 0.23, 0.25, 0.31, 0.38, 0.48])
cy = np.array([1.6, 1.0, 0.73, 0.49, 0.24, 0., -0.24, -0.49, -0.73, -1.0, -1.6])

c_mm = np.array([0.6, 0., -0.6])
c_n = np.array([-0.6, 0., 0.6])

cl_table = np.vstack((aoa_forces, cl))
cd_table = np.vstack((aoa_forces, cd))
cy_table = np.vstack((aoa_forces, cy))

cmm_table = np.vstack((aoa_moments, c_mm))
cn_table = np.vstack((aoa_moments, c_n))

# initial velocity
v_init = np.array([0.0, -0.0, -0.1])  # C-frame

# specific impulses
isp_stage1 = 360.
isp_stage2 = 390.

thrust_max_stage1 = 1.7e7
thrust_max_stage2 = 5.0e6
mass_flow_stage1 = thrust_max_stage1 / (gravity * isp_stage1)
mass_flow_stage2 = thrust_max_stage2 / (gravity * isp_stage2)

# compute times for stages
t_end_stage1 = fuel_cap_stage1 / mass_flow_stage1
duration_coast = 96.79
t_end_coast = t_end_stage1 + duration_coast
t_end_stage2 = t_end_stage1 + duration_coast + fuel_cap_stage2 / mass_flow_stage2
err = 1e-15
# make csv files

# create numpy arrays
inertia_csv = np.array([[0., *inertia_stage1_start, 0., 0., 0.],
                    [t_end_stage1, *inertia_stage1_end, 0., 0., 0.],
                        [t_end_coast, *inertia_stage1_end, 0., 0., 0.],
                        [t_end_coast+err, *inertia_stage2_start, 0., 0., 0.],
                        [t_end_stage2, *inertia_stage2_end, 0., 0., 0.]])

cg_csv = np.array([[0., *cg_stage1_start],
                   [t_end_stage1, *cg_stage1_end],
                   [t_end_coast, *cg_stage1_end],
                   [t_end_coast+err, *cg_stage2_start],
                   [t_end_stage2, *cg_stage2_end]])

mass_csv = np.array([[0., mass[0]],
                     [t_end_stage1, mass[1]],
                     [t_end_coast, mass[1]],
                     [t_end_coast+err, mass[2]],
                     [t_end_stage2, mass[3]]])

mrc_csv = np.array([[0., *mrc_stage1],
                    [t_end_stage1, *mrc_stage1],
                    [t_end_coast, *mrc_stage1],
                    [t_end_coast+err, *mrc_stage2],
                    [t_end_stage2, *mrc_stage2]])

# CL, CY, CD, CLL, CM, CN
aero_force_csv = np.vstack((cd_table, cy_table[1,:], cl_table[1,:])).T
aero_moment_csv = np.vstack((aoa_moments, np.zeros(len(aoa_moments)), cmm_table[1, :], cn_table[1, :])).T

np.savetxt(results_dir_path / "inertia_A17.csv", inertia_csv, delimiter=',')
np.savetxt(results_dir_path / "cg_A17.csv", cg_csv, delimiter=',')
np.savetxt(results_dir_path / "mass_A17.csv", mass_csv, delimiter=',')
np.savetxt(results_dir_path / "aero_force_A17.csv", aero_force_csv, delimiter=',')
np.savetxt(results_dir_path / "aero_moment_A17.csv", aero_moment_csv, delimiter=',')
np.savetxt(results_dir_path / "mrc_A17.csv", mrc_csv, delimiter=',')



