import numpy as np

GravG = 6.707e-39
rho_critical = 1.0537e-5 * 0.67**2.
Hubble_Now = 0.67 / 9.777752 # in Gyr


def default_cosmo_omega():
    omega_cdm = 0.258
    omega_b = 0.0484
    omega_g = 5.38e-5
    omega_L = 0.692
    omega_nu = 0.
    hubble = 0.67 / 9.777752
    return omega_cdm, omega_b, omega_g, omega_L, omega_nu, hubble
