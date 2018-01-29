import numpy as np
import os
from scipy.integrate import odeint
from constants import *
from scipy.interpolate import interp1d
from scipy.integrate import quad

path = os.getcwd()

class Single_FRW(object):

    def __init__(self, omega_b, omega_cdm, omega_L, omega_g, H_0):
        self.omega_b = omega_b
        self.omega_cdm = omega_cdm
        self.omega_L = omega_L
        self.omega_g = omega_g
        self.H_0 = H_0

    def Hubble(self, a):
        hubble = self.H_0 * np.sqrt(self.omega_g/a**4 + (self.omega_b+self.omega_cdm)/a**3
                                    + self.omega_L)
        return hubble

#    def solver(self, y, t):
#        dydt = y*self.Hubble(y)
#        return dydt
#
    def solver(self, y, t):
        dydt = y*self.H_0*np.sqrt(self.omega_g/y**4. + (self.omega_b + self.omega_cdm)/y**3.+ self.omega_L)
        return dydt
#    def solver(self, y, t):
#        a, yvar = y
#        dydt = [yvar, -a/2.*self.H_0**2.*(2.*self.omega_g/a**4. + (self.omega_b + self.omega_cdm)/a**3. -2.* self.omega_L)]
#        return dydt

    def obtain_scale_factor(self):
        n_tvals = 1000
        #time_arr = np.linspace(0., late_time, n_tvals)
        time_arr = np.logspace(-10, 1.5, n_tvals)
        #y0 = [1.]
        #y0 = [1e-12]
        #y0 = [1., self.H_0]
        #sol = odeint(self.solver, y0, time_arr)
        sol = np.zeros(n_tvals)
        for i in range(n_tvals):
            sol[i] = quad(lambda x: 1./2.234e-4/np.sqrt(5.38e-5/x**2.+(0.258 + 0.0484)/x+0.6936*x**2.), 0., time_arr[i])[0]
        np.savetxt(path + '/precomputed/SingleUniverse_FRW_scalefactor_Gyr.dat', np.column_stack((time_arr, sol)))
        #solinterp = interp1d(np.log10(time_arr), np.log10(sol.flatten()), kind='linear', bounds_error=False, fill_value=0.)
        conformal_time = np.zeros(n_tvals)
        for i in range(len(conformal_time)):
            conformal_time[i] = quad(lambda x: 1./2.234e-4/np.sqrt(5.38e-5+(0.258 + 0.0484)*x+0.6936*x**4.), 0., sol[i])[0] # units Mpc
            #conformal_time[i] = quad(lambda x: 1./2.234e-4/np.sqrt(5.38e-5+(0.258 + 0.0484)*x+0.7*x**4.), 0., sol.flatten()[i])[0] # units Mpc
        np.savetxt(path + '/precomputed/SingleUniverse_FRW_ConformalTime.dat', np.column_stack((time_arr, conformal_time)))
        
        
        return
