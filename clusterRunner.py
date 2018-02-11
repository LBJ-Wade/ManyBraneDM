import numpy as np
import os
from CMB import *
from multiprocessing import Pool

Ftag = 'StandardUniverse'
lmax_Pert = 5

compute_LP = True
compute_TH = True
compute_CMB = True
compute_MPS = True

kmin = 1e-3
kmax = 0.5
knum = 400
lmax = 2500
lvals = 250

OM_b = 0.0484
OM_c = 0.258 
OM_g = 5.38e-5
OM_L = 0.7 # Doesnt matter, it calculates with flat Uni


kgrid = np.logspace(np.log10(kmin), np.log10(kmax), knum)

def CMB_wrap(kval):
    SetCMB.runall(kVAL=kval, compute_LP=compute_LP, compute_TH=compute_TH,
               compute_CMB=False, compute_MPS=False)
    return

SetCMB = CMB(OM_b, OM_c, OM_g, OM_L, kmin=kmin, kmax=kmax, knum=knum, lmax=lmax,
             lvals=lvals, Ftag=Ftag, lmax_Pert=lmax_Pert)

if compute_LP or compute_TH:
    pool = Pool(processes=None)
    pool.map(CMB_wrap, kgrid)
    if compute_TH:
        SetCMB.SaveThetaFile()

if compute_CMB:
    SetCMB.runall(compute_LP=False, compute_TH=False,
               compute_CMB=compute_CMB, compute_MPS=False)
if compute_MPS:
    SetCMB.runall(compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=compute_MPS)

