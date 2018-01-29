import numpy as np
import os
from boltzmann import *

path = os.getcwd()






class CMB(object):

    def __init__(self, OM_b, OM_c, OM_g, OM_L, kmin=1e-5, kmax=10., knum=6,
                 compute_LP=False, compute_TH=False):
        self.OM_b = OM_b
        self.OM_c = OM_c
        self.OM_g = OM_g
        self.OM_L = OM_L
        self.OM_nu = (7./8)*(4./11.)**(4./3)*3.04 * OM_g
        self.kmin = kmin
        self.kmax = kmax
        self.knum = knum
        
        if compute_LP:
            self.kspace_linear_pert()
        return

    def kspace_linear_pert(self):
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        for k in kgrid:
            stepsize = 0.1
            success = False
            while not success:
                print 'Working on k = {:.3e}, step size = {:.3e}'.format(k, stepsize)
                try:
                    SingleUni = Universe(k, self.OM_b, self.OM_c, self.OM_g, self.OM_L, self.OM_nu,
                                         stepsize=stepsize).solve_system()
                    success = True
                except ValueError:
                    stepsize /= 10.
    
        print 'All k values computed!'
        return
