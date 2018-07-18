import numpy as np
from boltzmann import *
from frw_metric import *
from CMB import *

k = 0.1
omega_b = 0.0484
omega_cdm = 0.258
omega_r = 5.38e-5

#a = Single_FRW(0.0484, 0.258, 0.7, 5.38e-5,0.67/9.777752)
#a.obtain_scale_factor()

SingleUni = Universe(k, omega_b, omega_cdm, omega_r, 0.7, omega_r*3.04*(7./8)*(4./11)**(4./3), stepsize=0.01, lmax=10, testing=True, accuracy=1e-2)
SingleUni.clearfiles()
SingleUni.tau_functions()
SingleUni.solve_system()


#Nbranes = 1e7
#MultiUni = ManyBrane_Universe(Nbranes, k, [omega_b, omega_cdm/Nbranes], [0., 0.], [omega_r, 1e-6*(omega_r/omega_b)*(omega_cdm/Nbranes)], 0.7, [omega_r*3.04*(7./8)*(4./11)**(4./3), 1e-6*(omega_r/omega_b)*(omega_cdm/Nbranes)*3.04*(7./8)*(4./11)**(4./3)], stepsize=0.01, lmax=10, accuracy=1e-3, testing=True)
#MultiUni.clearfiles()
#MultiUni.solve_system()
#MultiUni.tau_functions()

#CMB(0.0484, 0.258, 5.38e-5, 0.7,kmin=1e-3, kmax=1e-1, knum=10, kVAL=1e-1, compute_LP=True, compute_TH=True)
#CMB(0.0484, 0.258, 5.38e-5, 0.7, kmin=5e-3, kmax=5e-1, knum=2, compute_LP=True, compute_TH=True)
