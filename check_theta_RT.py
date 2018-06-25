import numpy as np
import os
import glob

path = os.getcwd()
files = path + '/OutputFiles/*_ThetaFile_kval*.dat'

ell_indx = 0
endarr = []

file_list = glob.glob(files)
for ff in file_list:
    kval = float(ff[ff.find('_kval_')+6:ff.find('.dat')])
    loadf = np.loadtxt(ff)[ell_indx]
    endarr.append([kval, loadf])
    
finArr = np.asarray(endarr)[np.argsort(np.asarray(endarr[:,0]))
np.savetxt(path + '/OutputFiles/CHECK_THETA.dat')
