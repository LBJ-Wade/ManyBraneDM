import numpy as np
import os
from boltzmann import *
from frw_metric import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from multiprocessing import Pool
import glob
path = os.getcwd()


class CMB(object):

    def __init__(self, OM_b, OM_c, OM_g, OM_L, kmin=5e-3, kmax=0.5, knum=200,
                 lmax=2500, lvals=250,
                 Ftag='StandardUniverse', lmax_Pert=5):
        self.OM_b = OM_b
        self.OM_c = OM_c
        self.OM_g = OM_g
        self.OM_L = OM_L
        self.OM_nu = (7./8)*(4./11.)**(4./3)*3.04 * OM_g
        self.kmin = kmin
        self.kmax = kmax
        self.knum = knum
        self.Ftag = Ftag
        self.lmax = lmax
        self.lvals = lvals
        self.H_0 = 2.2348e-4 # units Mpc^-1
        self.lmax_Pert = lmax_Pert
        self.lmin = 10
        
        
        self.eta0 = 1.4100e4
        self.init_pert = -1/6.
        
        ell_val = range(self.lmin, self.lmax, 10)
        
        self.ThetaFile = path + '/OutputFiles/' + self.Ftag + '_ThetaCMB_Table.dat'
        self.ThetaTabTot = np.zeros((self.knum+1, len(ell_val)))
        self.ThetaTabTot[0,:] = ell_val

        self.fill_inx = 0
        
        
    
    
    def runall(self, kVAL=None, compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=False):
        
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        if compute_LP:
            print 'Computing Perturbation Fields...\n'
            self.kspace_linear_pert(kVAL)
        
        if compute_TH:
            print 'Computing Theta Files...\n'
            self.loadfiles()
            if kVAL is not None:
                self.theta_integration(kVAL, kVAL=kVAL)
            else:
                for k in kgrid:
                    self.theta_integration(k)

        if compute_CMB:
            print 'Computing CMB...\n'
            self.computeCMB()
        if compute_MPS:
            print 'Computing Matter Power Spectrum...\n'
            self.MatterPower()
        return
    
    def loadfiles(self):
        opt_depthL = np.loadtxt(path + '/precomputed/expOpticalDepth.dat')
        self.opt_depth = interp1d(np.log10(opt_depthL[:,0]), opt_depthL[:,1], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        time_table = np.loadtxt(path+'/precomputed/Times_Tables.dat')
        self.ct_to_scale = interp1d(np.log10(time_table[:,2]), np.log10(time_table[:,1]), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(time_table[:,1]), np.log10(time_table[:,2]), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        visfunc = np.loadtxt(path + '/precomputed/VisibilityFunc.dat')
        visfunc = np.log10(visfunc[visfunc[:,1] > 0.])
        self.Vfunc = interp1d(visfunc[:,0], visfunc[:,1], kind='linear',
                                  bounds_error=False, fill_value=-100.)
        self.eta_start = 10.**self.scale_to_ct(visfunc[-1,0])
        return

    def kspace_linear_pert(self, kVAL=None):
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        if kVAL is not None:
            kgrid = [kVAL]
        for k in kgrid:
            fileName = path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k)
            if os.path.isfile(fileName):
                continue
            stepsize = 0.1
            success = False
            while not success:
                print 'Working on k = {:.3e}, step size = {:.3e}'.format(k, stepsize)
                try:
                    SingleUni = Universe(k, self.OM_b, self.OM_c, self.OM_g, self.OM_L, self.OM_nu,
                                         stepsize=stepsize, accuracy=1e-3, lmax=self.lmax_Pert).solve_system()
                    success = True
                except ValueError:
                    stepsize /= 2.
    
        print 'All k values computed!'
        return

    def theta_integration(self, k, kVAL=None):
        if kVAL is not None:
            kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
            index = np.where(kgrid == kVAL)[0][0] + 1
        ell_tab = self.ThetaTabTot[0,:]
        
        fields = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k))
        theta0 = fields[:,6]
        psi = fields[:,-1]
        vb = fields[:,5]
        
        #theta0 = interp1d(np.log10(fields[:,0]), fields[:, 6], kind='linear', bounds_error=False, fill_value=0.)
        #psi = interp1d(np.log10(fields[:,0]), fields[:, -1], kind='linear', bounds_error=False, fill_value=0.)
        #vb = interp1d(np.log10(fields[:,0]), fields[:, 5], kind='linear', bounds_error=False, fill_value=0.)
        phi_dot = interp1d(np.log10(fields[1:,0]), np.diff(fields[:, 1])/np.diff(fields[:,0]), kind='linear', bounds_error=False, fill_value=0.)
        psi_dot = interp1d(np.log10(fields[1:,0]), np.diff(fields[:, -1])/np.diff(fields[:,0]), kind='linear', bounds_error=False, fill_value=0.)

        thetaVals = np.zeros(len(ell_tab))
        e_vals = fields[:,0]
        
        for i,ell in enumerate(ell_tab):
            # Using Eq 8.55 Dodelson
            jvalL = spherical_jn(int(ell), k*(self.eta0 - e_vals))
            diffJval = spherical_jn(int(ell), k*(self.eta0 - e_vals), derivative=True)
           
            term1 = np.trapz(self.visibility(e_vals)*(theta0 + psi)*jvalL , e_vals)
            term2 = -np.trapz(self.visibility(e_vals)*vb/k*diffJval, e_vals)
            term3 = np.trapz(self.exp_opt_depth(e_vals)* (psi_dot(np.log10(e_vals)) - phi_dot(np.log10(e_vals)))*jvalL, e_vals)
            thetaVals[i] = term1 + term2 + term3
        np.savetxt(path + '/OutputFiles/' + self.Ftag + '_ThetaFile_kval_{:.4e}.dat'.format(k), thetaVals)
        return
    
    def SaveThetaFile(self):
        if os.path.isfile(self.ThetaFile):
            os.remove(self.ThetaFile)
        ThetaFiles = glob.glob(path + '/OutputFiles/' + self.Ftag + '_ThetaFile_kval_*.dat')
        for i in range(len(ThetaFiles)):
            dat = np.loadtxt(ThetaFiles[i])
            self.ThetaTabTot[i+1,:] = dat
            os.remove(ThetaFiles[i])
        np.savetxt(self.ThetaFile, self.ThetaTabTot)

    def computeCMB(self):
        ThetaFile = path + '/OutputFiles/' + self.Ftag + '_ThetaCMB_Table.dat'
        thetaTab = np.loadtxt(ThetaFile)
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        ell_tab = self.ThetaTabTot[0,:]
        CL_table = np.zeros((len(ell_tab), 2))
        GF = ((self.OM_b+self.OM_c) / self.growthFactor(1.))**2.
        for i,ell in enumerate(ell_tab):
            cL = np.trapz(100.*np.pi/(9.*kgrid)*np.abs(thetaTab[1:, i]/self.init_pert)**2. , kgrid)
            CL_table[i] = [ell, ell*(ell+1)/(2.*np.pi)*cL*GF]

        np.savetxt(path + '/OutputFiles/' + self.Ftag + '_CL_Table.dat', CL_table)
        return

    def growthFactor(self, a):
        # D(a)
        Uni = Single_FRW(self.OM_b, self.OM_c, self.OM_L, self.OM_g, self.H_0)
        prefac = 5.*(self.OM_b + self.OM_c)/2. *(Uni.Hubble(a) / self.H_0) * self.H_0**3.
    
        integ_pt = quad(lambda x: 1./(x*Uni.Hubble(x)**3.), 0., a)[0]
        return prefac * integ_pt

    def exp_opt_depth(self, eta):
        aval = 10.**self.ct_to_scale(np.log10(eta))
        return self.opt_depth(np.log10(aval))
    
    def visibility(self, eta):
        ln10aval = self.ct_to_scale(np.log10(eta))
        return 10.**self.Vfunc(ln10aval)

    def MatterPower(self):
        # T(k) = \Phi(k, a=1) / \Phi(k = Large, a= 1)
        # P(k,a=1) = 2 pi^2 * \delta_H^2 * k / H_0^4 * T(k)^2
        Tktab = self.TransferFuncs()
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        PS = np.zeros_like(kgrid)
        for i,k in enumerate(kgrid):
            PS[i] = k*Tktab[i]**2.
        np.savetxt(path + '/OutputFiles/' + self.Ftag + '_MatterPowerSpectrum.dat', np.column_stack((kgrid, PS)))
        return

    def TransferFuncs(self):
        Minfields = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(self.kmin))
        LargeScaleVal = Minfields[-1, 1]
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        Tktab = np.zeros_like(kgrid)
        for i,k in enumerate(kgrid):
            field =  np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k))
            Tktab[i] = field[-1,1] / LargeScaleVal
        return Tktab


