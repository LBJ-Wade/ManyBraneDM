import numpy as np
from boltzmann import *
from frw_metric import *
from CMB import *

#a = Single_FRW(0.0484,0.258, 0.7, 5.38e-5,0.67/9.777752)
#a.obtain_scale_factor()

#SingleUni = Universe(1e-5, 0.0484, 0.258, 5.38e-5, 0.7, 5.38e-5*(7./8)*(4./11.)**(4./3)*3.04)
#SingleUni.solve_system()


CMB(0.0484, 0.258, 5.38e-5, 0.7,kmin=1e-5, kmax=10., knum=6, compute_LP=True, compute_TH=False)
