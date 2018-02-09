import numpy as np
from boltzmann import *
from frw_metric import *
from CMB import *

#a = Single_FRW(0.0484, 0.258, 0.7, 5.38e-5,0.67/9.777752)
#a.obtain_scale_factor()

#SingleUni = Universe(1.0155e-01, 0.022032/(0.67556)**2., 0.12038/(0.67556)**2., 2.473e-5/(0.67556)**2., 0.7, 2.473e-5/(0.67556)**2.*3.04*(7./8)*(4./11)**(4./3), stepsize=0.01, lmax=20)
#SingleUni.solve_system()
MultiUni = ManyBrane_Universe(0.1, [0.0484, 0.0484/10.], [0.258, 0.258+0.0484/10.*9], [5.38e-5, 5.38e-5/10.], 0.7, [5.38e-5*3.04*(7./8)*(4./11)**(4./3), 5.38e-5/10.*3.04*(7./8)*(4./11)**(4./3)], stepsize=0.01, lmax=5)
MultiUni.solve_system()


#CMB(0.0484, 0.258, 5.38e-5, 0.7,kmin=1e-3, kmax=1e-1, knum=10, kVAL=1e-1, compute_LP=True, compute_TH=True)
#CMB(0.0484, 0.258, 5.38e-5, 0.7, kmin=5e-3, kmax=5e-1, knum=2, compute_LP=True, compute_TH=True)
