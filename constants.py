import numpy as np
from scipy.interpolate import interp1d

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


def Yp_Prime(etaprime):
    betaL = np.logspace(-11, -4, 20)
    etaPR_Tab = np.array([8.641e-2, 1.498e-1, 2.015e-1, 2.265e-1, 2.390e-1, 2.478e-1,
                          2.554e-1, 2.627e-1, 2.698e-1, 2.768e-1, 2.83e-1, 2.909e-1,
                          2.984e-1, 3.060e-1, 3.14e-1, 3.22e-1, 3.31e-1, 3.39e-1,
                          3.485e-1, 3.61e-1])
    intrpTb = interp1d(np.log10(betaL), np.log10(etaPR_Tab), kind='linear', fill_value='extrapolate', bounds_error=False)
    return 10.**intrpTb(np.log10(etaprime))
