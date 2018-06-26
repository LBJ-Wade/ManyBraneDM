import numpy as np
import os
from boltzmann import *
from frw_metric import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from scipy.optimize import minimize
from multiprocessing import Pool
import glob
import math
path = os.getcwd()


class CMB(object):

    def __init__(self, OM_b, OM_c, OM_g, OM_L, kmin=5e-3, kmax=0.5, knum=200,
                 lmax=2500, lvals=250,
                 Ftag='StandardUniverse', lmax_Pert=5, multiverse=False,
                 OM_b2=0., OM_c2=0., OM_g2=0., OM_L2=0., Nbrane=0):
        
        self.OM_b = OM_b
        self.OM_c = OM_c
        self.OM_g = OM_g
        self.OM_L = OM_L
        self.OM_nu = (7./8)*(4./11.)**(4./3)*3.04 * OM_g

        self.OM_b2 = OM_b2
        self.OM_c2 = OM_c2
        self.OM_g2 = OM_g2
        self.OM_L2 = OM_L2
        self.OM_nu2 = (7./8)*(4./11.)**(4./3)*3.04 * OM_g2
        self.Nbrane = Nbrane
        if OM_b2 != 0.:
            self.PressFac = (OM_g2 / OM_b2) / (OM_g / OM_b)
        
        self.kmin = kmin
        self.kmax = kmax
        self.knum = knum
        self.Ftag = Ftag
        self.lmax = lmax
        self.lvals = lvals
        self.H_0 = 2.2348e-4 # units Mpc^-1
        self.lmax_Pert = lmax_Pert
        self.lmin = 10
        
        self.multiverse=multiverse
        
        self.eta0 = 1.4100e4
        self.init_pert = -1/6.
        
        ell_val = range(self.lmin, self.lmax, 15)
        
        self.ThetaFile = path + '/OutputFiles/' + self.Ftag + '_ThetaCMB_Table.dat'
        self.ThetaTabTot = np.zeros((self.knum+1, len(ell_val)))
        self.ThetaTabTot[0,:] = ell_val

        self.fill_inx = 0
    
    def runall(self, kVAL=None, compute_LP=False, compute_TH=False,
               compute_CMB=False, compute_MPS=False):
        
        if compute_MPS:
            self.kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        else:
            #self.kgrid = np.linspace(self.kmin, self.kmax, self.knum)
            self.kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        
        if compute_LP:
            print 'Computing Perturbation Fields...\n'
            self.kspace_linear_pert(kVAL)
        
        if compute_TH:
            print 'Computing Theta Files...\n'
            self.loadfiles()
            if kVAL is not None:
                self.theta_integration(kVAL, kVAL=kVAL)
            else:
                for k in self.kgrid:
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
        self.opt_depth = interp1d(np.log10(opt_depthL[:,0]), opt_depthL[:,1], kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
        time_table = np.loadtxt(path+'/precomputed/Times_Tables.dat')
        self.ct_to_scale = interp1d(np.log10(time_table[:,2]), np.log10(time_table[:,1]), kind='cubic',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(time_table[:,1]), np.log10(time_table[:,2]), kind='cubic',
                                    bounds_error=False, fill_value='extrapolate')
        visfunc = np.loadtxt(path + '/precomputed/VisibilityFunc.dat')
        visfunc = np.log10(visfunc[visfunc[:,1] > 0.])
        self.Vfunc = interp1d(visfunc[:,0], visfunc[:,1], kind='cubic',
                                  bounds_error=False, fill_value=-100.)
        self.eta_start = 10.**self.scale_to_ct(visfunc[-1,0])
    
        return

    def kspace_linear_pert(self, kVAL=None):
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        
        if kVAL is not None:
            kgrid = [kVAL]
        else:
            kgrid = self.kgrid
        for k in kgrid:
            if self.multiverse:
                fileName = path + '/OutputFiles/' + self.Ftag + \
                         '_FieldEvolution_{:.4e}_Nbrane_{:.0e}_PressFac_{:.2e}.dat'.format(k, self.Nbrane,
                                                                                           self.PressFac)
            else:
                fileName = path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k)
            if os.path.isfile(fileName):
                continue
            stepsize = 0.005
            success = False
            while not success:
                print 'Working on k = {:.3e}, step size = {:.3e}'.format(k, stepsize)
                try:
                    if not self.multiverse:
                        SingleUni = Universe(k, self.OM_b, self.OM_c, self.OM_g, self.OM_L, self.OM_nu,
                                             stepsize=stepsize, accuracy=1e-4, lmax=self.lmax_Pert).solve_system()
                    else:
                        ManyBrane_Universe(self.Nbrane, k, [self.OM_b, self.OM_b2], [self.OM_c, self.OM_c2],
                                          [self.OM_g, self.OM_g2], [self.OM_L, self.OM_L2],
                                          [self.OM_nu, self.OM_nu2], accuracy=1e-3,
                                          stepsize=stepsize, lmax=self.lmax_Pert).solve_system()
                    success = True
                except ValueError:
                    stepsize /= 2.
    
        print 'All k values computed!'
        return

    def theta_integration(self, k, kVAL=None):
        filename = path + '/OutputFiles/' + self.Ftag + '_ThetaFile_kval_{:.4e}.dat'.format(k)
        if os.path.isfile(filename):
            return
        if kVAL is not None:
            #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
            index = np.where(self.kgrid == kVAL)[0][0] + 1
        ell_tab = self.ThetaTabTot[0,:]
        #eta_full = np.logspace(2, np.log10(self.eta0), 4000)
        
        fields = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k))
        
        theta0_I = interp1d(np.log10(fields[:,0]), fields[:, 6], kind='cubic', bounds_error=False, fill_value=0.)
        theta1_I = interp1d(np.log10(fields[:,0]), fields[:, 9], kind='cubic', bounds_error=False, fill_value=0.)
        psi_I = interp1d(np.log10(fields[:,0]), fields[:, -1], kind='cubic', bounds_error=False, fill_value=0.)
        vb_I = interp1d(np.log10(fields[:,0]), fields[:, 5], kind='cubic', bounds_error=False, fill_value=0.)
        
        PiPolar = interp1d(np.log10(fields[:,0]), fields[:, 6] + fields[:, 11] + fields[:, 12], kind='cubic', bounds_error=False, fill_value=0.)
        pre_2nd_derTerm = (fields[:, 6] + fields[:, 11] + fields[:, 12])*self.visibility(fields[:,0])
        sec_DerTerm = np.zeros(len(pre_2nd_derTerm) - 2)
        
        etaVisMax = self.vis_max_eta()
        
        for i in range(len(pre_2nd_derTerm) - 2):
            h2 = fields[i+2,0] - fields[i+1, 0]
            h1 = fields[i+1,0] - fields[i, 0]
            sec_DerTerm[i] = 2.*(h2*pre_2nd_derTerm[i+2] - (h1+h2)*pre_2nd_derTerm[i+1] + h1*pre_2nd_derTerm[i])/(h1*h2*(h1+h2))
        DerTerm = interp1d(np.log10(fields[:,0][1:-1]), sec_DerTerm, kind='cubic', bounds_error=False, fill_value=0.)

        phi_dot = interp1d(np.log10(fields[1:,0]), np.diff(fields[:, 1])/np.diff(fields[:,0]), kind='cubic', bounds_error=False, fill_value=0.)
        psi_dot = interp1d(np.log10(fields[1:,0]), np.diff(fields[:, -1])/np.diff(fields[:,0]), kind='cubic', bounds_error=False, fill_value=0.)

        thetaVals = np.zeros(len(ell_tab))
#        testINTS = np.zeros((len(ell_tab), 3))

        e_vals = fields[:,0]
        
        for i,ell in enumerate(ell_tab):
            term1 = (theta0_I(np.log10(etaVisMax)) + psi_I(np.log10(etaVisMax)))* spherical_jn(int(ell), k*(self.eta0 - etaVisMax))

            term2 = 3.*theta1_I(np.log10(etaVisMax))*(spherical_jn(int(ell-1), k*(self.eta0 - etaVisMax)) - (ell+1.)*spherical_jn(int(ell), k*(self.eta0 - etaVisMax))/(k*(self.eta0 - etaVisMax)))
#            term3 = quad(lambda x:  self.exp_opt_depth(x)*(psi_dot(np.log10(x)) - phi_dot(np.log10(x)))*
#                           spherical_jn(int(ell), k*(self.eta0 - x)), self.eta_start, self.eta0, limit=200)[0]
#        
            thetaVals[i] = term1 + term2 #+ term3
#
#            if np.abs(thetaVals[i]) < 1e-50:
#                thetaVals[i] = 1e-50
#            testINTS[i] = [term1, term2, term3]

        np.savetxt(filename, thetaVals)
        #np.savetxt(path + '/OutputFiles/TESTING_TERMS_ThetaFile_kval_{:.4e}.dat'.format(k), testINTS)
        return
    
    def SaveThetaFile(self, test=False):
        #kgrid = np.linspace(self.kmin, self.kmax, self.knum)
        kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        if os.path.isfile(self.ThetaFile):
            os.remove(self.ThetaFile)
        ThetaFiles = glob.glob(path + '/OutputFiles/' + self.Ftag + '_ThetaFile_kval_*.dat')
        klist = np.array([])
        for i in range(len(ThetaFiles)):
            kval = float(ThetaFiles[i][ThetaFiles[i].find('kval_')+5:ThetaFiles[i].find('.dat')])
            klist = np.append(klist, kval)
        
        klist = np.sort(klist)
        for i in range(len(ThetaFiles)):
            dat = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_ThetaFile_kval_{:.4e}.dat'.format(klist[i]))
            self.ThetaTabTot[i+1,:] = dat
            os.remove(path + '/OutputFiles/' + self.Ftag + '_ThetaFile_kval_{:.4e}.dat'.format(klist[i]))
        np.savetxt(self.ThetaFile, self.ThetaTabTot, fmt='%.4e')
        
        if test:
            #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
            np.savetxt(path + '/OutputFiles/TESTING_THETA.dat', np.column_stack((kgrid, self.ThetaTabTot[1:,:])))
        return

    def computeCMB(self):
        ThetaFile = path + '/OutputFiles/' + self.Ftag + '_ThetaCMB_Table.dat'
        thetaTab = np.loadtxt(ThetaFile)
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        ell_tab = self.ThetaTabTot[0,:]
        CL_table = np.zeros((len(ell_tab), 2))
        GF = ((self.OM_b+self.OM_c) / self.growthFactor(1.))**2.
        for i,ell in enumerate(ell_tab):
#            cL_interp = interp1d(np.log10(self.kgrid), np.log10(100.*np.pi/(9.*self.kgrid)*np.abs(thetaTab[1:, i]/self.init_pert)**2.), kind='cubic', fill_value=-30)
#            cL_interp = interp1d(np.log10(self.kgrid), 100.*np.pi/(9.*self.kgrid)*np.abs(thetaTab[1:, i]/self.init_pert)**2., kind='cubic', fill_value=0.)
            # integrate in log k
            cL_interp = interp1d(np.log(self.kgrid), (thetaTab[1:, i]/self.init_pert), kind='cubic', fill_value=0.)
            CLint = quad(lambda x: 100.*np.pi/(9.)*cL_interp(x)**2., np.log(self.kgrid[0]), np.log(self.kgrid[-1]), limit=200)
#            CLint = quad(lambda x: cL_interp(np.log10(x)), self.kgrid[0], self.kgrid[-1], limit=200)
            CL_table[i] = [ell, ell*(ell+1)/(2.*np.pi)*CLint[0]*GF]
            if math.isnan(CLint[0]):
                print i, ell
                print np.abs(thetaTab[1:, i]/self.init_pert)**2.
                print thetaTab[1:, i]
                print cL_interp(np.log10(self.kgrid))
                exit()

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
    
    def vis_max_eta(self):
        etaL = np.logspace(-1, np.log10(self.eta0), 2000)
        visEval = self.visibility(etaL)
        return etaL[np.argmax(visEval)]

    def MatterPower(self):
        # T(k) = \Phi(k, a=1) / \Phi(k = Large, a= 1)
        # P(k,a=1) = 2 pi^2 * \delta_H^2 * k / H_0^4 * T(k)^2
        Tktab = self.TransferFuncs()
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        PS = np.zeros_like(self.kgrid)
        for i,k in enumerate(self.kgrid):
            PS[i] = k*Tktab[i]**2.
        if self.multiverse:
            np.savetxt(path + '/OutputFiles/' + self.Ftag +
                       '_MatterPowerSpectrum_Nbrane_{:.0e}_PressFac_{:.2e}.dat'.format(self.Nbrane,self.PressFac),
                       np.column_stack((self.kgrid, PS)))
        else:
            np.savetxt(path + '/OutputFiles/' + self.Ftag + '_MatterPowerSpectrum.dat', np.column_stack((self.kgrid, PS)))
        return

    def TransferFuncs(self):
        if self.multiverse:
            Minfields = np.loadtxt(path + '/OutputFiles/' + self.Ftag +
                        '_FieldEvolution_{:.4e}_Nbrane_{:.0e}_PressFac_{:.2e}.dat'.format(self.kmin, self.Nbrane, self.PressFac))
        else:
            Minfields = np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(self.kmin))
        LargeScaleVal = Minfields[-1, 1]
        #kgrid = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.knum)
        Tktab = np.zeros_like(self.kgrid)
        for i,k in enumerate(self.kgrid):
            if self.multiverse:
                field =  np.loadtxt(path + '/OutputFiles/'+ self.Ftag +
                                    '_FieldEvolution_{:.4e}_Nbrane_{:.0e}_PressFac_{:.2e}.dat'.format(k, self.Nbrane, self.PressFac))
            else:
                field =  np.loadtxt(path + '/OutputFiles/' + self.Ftag + '_FieldEvolution_{:.4e}.dat'.format(k))
            Tktab[i] = field[-1,1] / LargeScaleVal
        return Tktab


