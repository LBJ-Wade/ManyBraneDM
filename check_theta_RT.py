import numpy as np
import os
import glob

path = os.getcwd()

inRUN = False

if inRUN:
    files = path + '/OutputFiles/*_ThetaFile_kval*.dat'
else:
    files = path + '/OutputFiles/StandardUniverse_ThetaCMB_Table.dat'

ell_indx = 0
endarr = []

file_list = glob.glob(files)
for ff in file_list:
    if inRUN:
        kval = float(ff[ff.find('_kval_')+6:ff.find('.dat')])
        loadf = np.loadtxt(ff)[ell_indx]
        endarr.append([kval, loadf])
    else:
        finArr = np.loadtxt(ff)[1:, ell_indx]

if inRUN:
    finArr = np.asarray(endarr)[np.argsort(np.asarray(endarr)[:,0])]

np.savetxt(path + '/OutputFiles/CHECK_THETA.dat', finArr)
