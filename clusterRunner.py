import numpy as np
import os
from CMB import *

Ftag = 'StandardUniverse'
lmax_Pert = 5

compute_LP = True
compute_TH = True
compute_CMB = True
compute_MPS = True

kmin = 1e-3
kmax = 0.5
knum = 200
lmax = 2500
lval = 250

OM_b = 0.0484
OM_c = 0.258 
OM_g = 5.38e-5
OM_L = 0.7 # Doesnt matter, it calculates with flat Uni


kgrid = np.logspace(np.log10(kmin), np.log10(kmax), knum)

if compute_LP or compute_TH:
    for k in kgrid:
        CMB(OM_b, OM_c, OM_g, OM_L, kmin=kmin, kmax=kmax, knum=knum, lmax=lmax, 
            lvals=lvals, compute_LP=True, compute_TH=True, 
            compute_CMB=False, compute_MPS=False, kVAL=k, 
            Ftag=Ftag, lmax_Pert=lmax_Pert)

if compute_CMB:
    CMB(OM_b, OM_c, OM_g, OM_L, kmin=kmin, kmax=kmax, knum=knum, lmax=lmax,
        lvals=lvals, compute_LP=False, compute_TH=False, compute_CMB=True,
        compute_MPS=False, kVAL=k, Ftag=Ftag, lmax_Pert=lmax_Pert)
if compute_MPS:
    CMB(OM_b, OM_c, OM_g, OM_L, kmin=kmin, kmax=kmax, knum=knum, lmax=lmax,
        lvals=lvals, compute_LP=False, compute_TH=False, compute_CMB=False,
        compute_MPS=True, kVAL=k, Ftag=Ftag, lmax_Pert=lmax_Pert)
