import numpy as np
import os
import sympy
#from sympy import *
#from sympy.matrices import *
import scipy
import scipy.linalg
from scipy.optimize import fsolve
from scipy.linalg import lu_solve, lu_factor, inv
from scipy.integrate import ode, quad, odeint
from scipy.interpolate import interp1d
from constants import *
import time

path = os.getcwd()


class Universe(object):

    def __init__(self, k, omega_b, omega_cdm, omega_g, omega_L, omega_nu, accuracy=1e-3,
                 stepsize=0.01, lmax=5, testing=False):
        self.omega_b = omega_b
        self.omega_cdm = omega_cdm
        self.omega_g = omega_g
        self.omega_nu = omega_nu
        self.omega_M = omega_cdm + omega_b
        self.omega_R = omega_g + omega_nu
        self.omega_L = 1. - self.omega_M - self.omega_R
        self.H_0 = 2.2348e-4 # units Mpc^-1

        self.Lmax = lmax
        self.stepsize = stepsize
        
        self.k = k
        print 'Solving perturbations for k = {:.3e} \n'.format(k)
        
        self.accuracy = accuracy
        self.TotalVars = 8 + 3*self.Lmax
        self.step = 0
        
        self.Theta_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Theta_P_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Neu_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        
        self.combined_vector = np.zeros(self.TotalVars ,dtype=object)
        self.Psi_vec = []
        self.combined_vector[0] = self.Phi_vec = []
        self.combined_vector[1] = self.dot_rhoCDM_vec = []
        self.combined_vector[2] = self.dot_velCDM_vec = []
        self.combined_vector[3] = self.dot_rhoB_vec = []
        self.combined_vector[4] = self.dot_velB_vec = []
        for i in range(self.Lmax + 1):
            self.combined_vector[5+i*3] = self.Theta_Dot[i] = []
            self.combined_vector[6+i*3] = self.Theta_P_Dot[i] = []
            self.combined_vector[7+i*3] = self.Neu_Dot[i] = []
        
        self.compute_funcs()
#        print 'Matter-Radiation Eq: ', (self.omega_M/self.omega_R - 1.)
#        print self.omega_M, self.omega_R
#        exit()

        self.testing = testing
        if self.testing:
            self.aLIST = []
            self.etaLIST = []
            self.csLIST = []
            self.hubLIST = []
            self.dtauLIST = []
            self.xeLIST = []
        
        return

    def compute_funcs(self, preload=True):
        a0_init = np.logspace(-14, 0, 1e4)
        eta_list = np.zeros_like(a0_init)
        for i in range(len(a0_init)):
            eta_list[i] = self.conform_T(a0_init[i])
        
        self.eta_0 = eta_list[-1]
        self.ct_to_scale = interp1d(np.log10(eta_list), np.log10(a0_init), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(a0_init), np.log10(eta_list), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        
        self.Thermal_sln()
        # DONT FORGET ABOUT THIS
        if preload:
            generic_full_files = np.loadtxt('precomputed/explanatory02_thermodynamics.dat')
            avals = 1./(1. + generic_full_files[:,0])
            visfunc = generic_full_files[:,5]
            exptau = generic_full_files[:,4]
            temb = generic_full_files[:,6]
            
            csoundb = generic_full_files[:,7]
            xe = generic_full_files[:,2]
            self.Cs_Sqr = interp1d(avals, csoundb, kind='linear', bounds_error=False, fill_value='extrapolate')
            np.savetxt(path + '/precomputed/tb_working.dat', np.column_stack((avals, temb)))
            np.savetxt(path + '/precomputed/xe_working.dat', np.column_stack((avals, xe)))
            np.savetxt(path + '/precomputed/working_VisibilityFunc.dat', np.column_stack((avals, visfunc)))
            np.savetxt(path + '/precomputed/working_expOpticalDepth.dat', np.column_stack((avals, exptau)))

    def clearfiles(self):
        if os.path.isfile(path + '/precomputed/xe_working.dat'):
            os.remove(path + '/precomputed/xe_working.dat')
        if os.path.isfile(path + '/precomputed/tb_working.dat'):
            os.remove(path + '/precomputed/tb_working.dat')

        if os.path.isfile(path + '/precomputed/working_expOpticalDepth.dat'):
            os.remove(path + '/precomputed/working_expOpticalDepth.dat')
        if os.path.isfile(path + '/precomputed/working_VisibilityFunc.dat'):
            os.remove(path + '/precomputed/working_VisibilityFunc.dat')

    def Thermal_sln(self):
        self.tb_fileNme = path + '/precomputed/tb_working.dat'
        self.Xe_fileNme = path + '/precomputed/xe_working.dat'
        if not os.path.isfile(self.tb_fileNme) or not os.path.isfile(self.Xe_fileNme):
#            tvals = np.linspace(5., -1, 500)
            tvals = np.linspace(3.4, -1, 500)
            y0 = [1.079, 2.7255 * (1. + 10.**tvals[0])]
#            y0 = [1., 2.7255 * (1. + 10.**tvals[0]), 0., 1.]
            val_sln = odeint(self.thermal_funcs, y0, tvals)
            avals = 1. / (1. + 10.**tvals)
            zreion = 12.
            tanhV = .5*(1. + 0.08112)*(1.+np.tanh(((1.+zreion)**(3./2.) - (1.+10.**tvals)**(3./2.)) / (3./2.)*np.sqrt(1.+zreion)*0.5))
            zreionHE = 3.5
            tanhV += .5*0.08112*(1.+np.tanh(((1.+zreionHE)**(3./2.) - (1.+10.**tvals)**(3./2.)) / (3./2.)*np.sqrt(1.+zreionHE)*0.5))
            val_sln[:,0] = np.maximum(val_sln[:,0], tanhV)
            self.Tb_drk = np.column_stack((avals, val_sln[:, 1]))
            np.savetxt(self.tb_fileNme, self.Tb_drk)
            self.Xe_dark = np.column_stack((avals, val_sln[:,0]))
            np.savetxt(self.Xe_fileNme, self.Xe_dark)
    
        else:
 
            self.Tb_drk = np.loadtxt(self.tb_fileNme)
            self.Xe_dark = np.loadtxt(self.Xe_fileNme)
        
        self.Tb = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), bounds_error=False, fill_value='extrapolate')
        self.Xe = interp1d(np.log10(self.Xe_dark[:,0]), np.log10(self.Xe_dark[:,1]), bounds_error=False, fill_value=np.log10(1.1622))
        return


    def thermal_funcs(self, val, z):
#        xe, T, xhe_1, xhe_2 = val
        xe, T = val
        return [self.xeDiff([xe], z, T)[0], self.dotT([T], z, xe)]
    
    def dotT(self, T, lgz, xe):
        kb = 8.617e-5/1e9 # Gev/K
        thompson_xsec = 6.65e-25 # cm^2
        aval = 1. / (1. + 10.**lgz)
        Yp = 0.245
        Mpc_to_cm = 3.086e24

        mol_wei = (0.5*(1.-Yp) + Yp*1.33)*xe + (1.*(1.-Yp) + Yp*4.)*np.abs(1.16-xe)

        n_b = 2.503e-7*(1.+10.**lgz)**3.
        hub = self.hubble(aval)
        omega_Rat = self.omega_g / self.omega_b
        jacF = - 1. * (10.**lgz * np.log(10.))
        return (-2.*T[0]*aval + (1./hub)*(8./3.)*(mol_wei/5.11e-4)*omega_Rat*(xe*n_b*thompson_xsec)*(2.7255*(1.+10.**lgz) - T[0])*Mpc_to_cm)*jacF
    
#    def Cs_Sqr(self, a):
#        kb = 8.617e-5/1e9 # GeV/K
#        facxe = 10.**self.Xe(np.log10(a))
#        Yp = 0.245
#
#        mol_wei = (0.5*(1.-Yp) + Yp*1.33)*facxe + (1.*(1.-Yp) + Yp*4.)*np.abs(1.16-facxe)
#        Tb = 10.**self.Tb(np.log10(a))
#        if a < 1:
#            lgZ = np.log10(1./a - 1.)
#        else:
#            lgZ = -10
#        extraPT = self.dotT([Tb], lgZ, facxe) *(-1./Tb)*(1.+10.**lgZ)/(np.log(10.) * 10.**lgZ)
#        val_r = 2.*kb*Tb/mol_wei*(1. - 1./3. * extraPT/Tb)
#        if val_r < 0.:
#            return np.abs(val_r)
#        if val_r > 1:
#            return 1.
#        return val_r

    def xeDiff(self, val, y, tgas, hydrogen=True, first=True):
        if y > 3.5:
            return [0.]
        yy = 10.**y
        ep0 = 13.6/1e9  # GeV
        kb = 8.617e-5/1e9 # Gev/K
        GeV_cm = 5.06e13
        Mpc_to_cm = 3.086e24
        me = 5.11e-4 # GeV
        aval = 1. / (1.+yy)
        Yp = 0.245
        
        n_b = 2.503e-7 / aval**3.
        hub = self.hubble(aval)
        FScsnt = 7.29e-3
        
        alpha2 = 9.78*(FScsnt/me)**2.*np.sqrt(ep0/(kb*tgas))*np.log(ep0/(kb*tgas))/(GeV_cm**2.) # cm^2
        beta = alpha2*(me*(kb*tgas)/(2.*np.pi))**(3./2.)*np.exp(-ep0/(kb*tgas))*GeV_cm**3. # 1/cm
        beta2 = alpha2*(me*(kb*tgas)/(2.*np.pi))**(3./2.)*np.exp(-ep0/(4.*kb*tgas))*GeV_cm**3.*Mpc_to_cm
        
        if val[0] > 0.999:
            Cr = 1.
        else:
            Lalpha = (3.*ep0)**3.*hub / (64.*np.pi**2*(1-val[0])*n_b*Yp) * GeV_cm**3.
            L2g = 8.227 / 2.998e10 * Mpc_to_cm
            Cr = (Lalpha + L2g) / (Lalpha + L2g + beta2)
        
        Value = Cr*np.log(10.)*yy*(-aval)/(hub)*((1.-val[0])*beta - val[0]**2.*n_b*alpha2)*Mpc_to_cm
        return [Value]
    
    
    def tau_functions(self):
        self.fileN_optdep = path + '/precomputed/working_expOpticalDepth.dat'
        self.fileN_visibil = path + '/precomputed/working_VisibilityFunc.dat'
        Mpc_to_cm = 3.086e24
        if not os.path.isfile(self.fileN_visibil) or not os.path.isfile(self.fileN_optdep):
            avals = np.logspace(-7, 0, 10000)
            Yp = 0.245
            n_b = 2.503e-7 / avals**3.
            thompson_xsec = 6.65e-25 # cm^2
            xevals = 10.**self.Xe(np.log10(avals))
            hubbs = self.hubble(avals)
            dtau = -xevals * (1. - Yp) * n_b * thompson_xsec * avals * Mpc_to_cm
            tau = np.zeros_like(dtau)
            etavals = np.zeros_like(avals)
            for i in range(len(avals)):
                etavals[i] = self.conform_T(avals[i])
            for i in range(len(dtau) - 1):
                tau[i+1] = np.trapz(-dtau[i:], etavals[i:])
            tau[0] = tau[1]
            np.savetxt(self.fileN_optdep, np.column_stack((avals, np.exp(-tau))))
            np.savetxt(self.fileN_visibil, np.column_stack((avals, -dtau * np.exp(-tau))))
    
        return

    def init_conds(self, eta_0, aval):
        OM = self.omega_M * self.H_0**2./self.hubble(aval)**2./aval**3.
        OR = self.omega_R * self.H_0**2./self.hubble(aval)**2./aval**4.
#        OR = self.omega_g * self.H_0**2./self.hubble(aval)**2./aval**4.
        ONu = self.omega_nu * self.H_0**2./self.hubble(aval)**2./aval**4.
        rfactor = ONu / (0.75*OM*aval + OR)
        HUB = self.hubble(aval)

        self.inital_perturb = -1./6.
        for i in range(1):
            self.Psi_vec.append(self.inital_perturb)
            self.Phi_vec.append(-(1.+2.*rfactor/5.)*self.Psi_vec[-1])
            self.dot_rhoCDM_vec.append(-3./2.*self.Psi_vec[-1])
            self.dot_velCDM_vec.append(1./2.*eta_0*self.k*self.Psi_vec[-1])
            self.dot_rhoB_vec.append(-3./2.*self.Psi_vec[-1])
            self.dot_velB_vec.append(1./2*eta_0*self.k*self.Psi_vec[-1])
            
            self.Theta_Dot[0].append(-1./2.*self.Psi_vec[-1])
            self.Theta_Dot[1].append(1./6.*eta_0*self.k*self.Psi_vec[-1])
            self.Neu_Dot[0].append(-1./2.*self.Psi_vec[-1])
            self.Neu_Dot[1].append(1./6.*eta_0*self.k*self.Psi_vec[-1])
            self.Neu_Dot[2].append(1./30.*(self.k*eta_0)**2.*self.Psi_vec[-1])
            

            for i in range(self.Lmax + 1):
                if i > 1:
                    self.Theta_Dot[i].append(0.)
                self.Theta_P_Dot[i].append(0.)
                if i > 2:
                    self.Neu_Dot[i].append(0.)
    
        self.step = 0
        return
    
    
    def solve_system(self):
        eta_st = np.min([1e-3/self.k, 1e-1/0.7]) # Initial conformal time in Mpc
        y_st = np.log(self.scale_a(eta_st))
        eta_st = self.conform_T(np.exp(y_st))
        
        self.init_conds(eta_st, np.exp(y_st))
        self.eta_vector = [eta_st]
        self.y_vector = [y_st]
        
##        #TESTING
#        aL = np.logspace(-9, 0, 300)
#        xeV = 10.**self.Xe(np.log10(aL))
#        csT = np.zeros_like(aL)
#        for i in range(len(csT)):
#            csT[i] = self.Cs_Sqr(aL[i])
#        tbT = 10.**self.Tb(np.log10(aL))
#        np.savetxt('TEST____.dat', np.column_stack((aL, xeV, csT, tbT, self.hubble(aL))))
#        exit()

        try_count = 0.
        try_max = 20.
        FailRUN = False
        last_step_up = False
        while (self.eta_vector[-1] < (self.eta_0-1.)):
            
            if try_count > try_max:
                #print 'FAIL TRY MAX....Breaking.'
                FailRUN=True
                break
            y_use = self.y_vector[-1] + self.stepsize
            eta_use = self.conform_T(np.exp(y_use))
            if (eta_use > self.eta_0):
                eta_use = self.eta_0
                y_use = np.log(self.scale_a(eta_use))
            self.eta_vector.append(eta_use)

            y_diff = y_use - self.y_vector[-1]
            self.y_vector.append(y_use)
            
            if self.step%3000 == 0:
                print 'Last a: {:.7e}, New a: {:.7e}'.format(np.exp(self.y_vector[-2]), np.exp(self.y_vector[-1]))
            if ((y_diff > eta_use*np.exp(y_use)*self.hubble(np.exp(y_use))) or
                (y_diff > np.max([np.exp(y_use)*self.hubble(np.exp(y_use)),
                                  np.exp(y_use)*self.hubble(np.exp(y_use))/self.k]))):
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                
                try_count += 1
                continue
            self.step_solver()
    
            test_epsilon = self.epsilon_test(np.exp(self.y_vector[-1]))
            if np.abs(test_epsilon) > self.accuracy and self.step > 10:
                raise ValueError
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                #print 'Failed epsilon test...   Value: {:.3e}'.format(test_epsilon)
                try_count += 1
                continue
            self.step += 1
            if (np.abs(test_epsilon) < 1e-4*self.accuracy) and not last_step_up:
                #self.stepsize *= 1.1
                self.stepsize *= 1.
                last_step_up = True
                #print 'Increase Step Size'
            else:
                last_step_up = False
            try_count = 0.
            
        if not FailRUN:
            print 'Saving File...'
            self.save_system()
        return

    def step_solver(self):
        if self.step > 0:
            tau_n = (self.y_vector[-1] - self.y_vector[-2]) / (self.y_vector[-2] - self.y_vector[-3])
        else:
            tau_n = (self.y_vector[-1] - self.y_vector[-2]) / self.y_vector[-2]
        
        delt = (self.y_vector[-1] - self.y_vector[-2])
        Ident = np.eye(self.TotalVars)
        Jmat = self.matrix_J(self.y_vector[-1])
        Amat = (1.+2.*tau_n)/(1.+tau_n)*Ident - delt*Jmat
        bvec = self.b_vector(tau_n)
        ysol = np.matmul(inv(Amat),bvec)
        for i in range(self.TotalVars):
            self.combined_vector[i].append(ysol[i])
        return
    
    def b_vector(self, tau):
        bvec = np.zeros(self.TotalVars)
        for i in range(self.TotalVars):
            if self.step == 0:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1]
            else:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1] - tau**2./(1.+tau)*self.combined_vector[i][-2]
        return bvec
    
    def matrix_J(self, z_val):
        a_val = np.exp(z_val)
        eta = self.conform_T(a_val)
        Jma = np.zeros((self.TotalVars, self.TotalVars))
        Rfac = (3.*self.rhoB(a_val))/(4.*self.rhoG(a_val))
        RR = (4.*self.rhoG(a_val))/(3.*self.rhoB(a_val))
        HUB = self.hubble(a_val)
        dTa = -10.**self.Xe(np.log10(a_val))*(1.-0.245)*2.503e-7*6.65e-29*1e4/a_val**2./3.24078e-25
        CsndB = self.Cs_Sqr(a_val)
        
        if self.testing:
            self.aLIST.append(a_val)
            self.etaLIST.append(eta)
            self.hubLIST.append(HUB)
            self.csLIST.append(CsndB)
            self.dtauLIST.append(dTa)
            self.xeLIST.append(10.**self.Xe(np.log10(a_val)))
        
        tflip_TCA = 1e-12
        
        PsiTerm = np.zeros(self.TotalVars)
        PsiTerm[0] += -1.
        PsiTerm[11] += -12.*(a_val/self.k)**2.*self.rhoG(a_val)
        PsiTerm[13] += -12.*(a_val/self.k)**2.*self.rhoNeu(a_val)
        
        # Phi Time derivative
        Jma[0,:] += PsiTerm
        Jma[0,0] += -((self.k/(HUB*a_val))**2.)/3.
        Jma[0,1] += 1./(HUB**2.*2.)*self.rhoCDM(a_val)
        Jma[0,3] += 1./(HUB**2.*2.)*self.rhoB(a_val)
        Jma[0,5] += 2./(HUB**2.)*self.rhoG(a_val)
        Jma[0,7] += 2./(HUB**2.)*self.rhoNeu(a_val)

        # CDM density
        Jma[1,2] += -self.k/(HUB*a_val)
        Jma[1,:] += -3.*Jma[0,:]

        # CDM velocity
        Jma[2,2] += -1.
        Jma[2,:] += self.k/(HUB*a_val)*PsiTerm

        # Baryon density
        Jma[3,4] += -self.k / (HUB*a_val)
        Jma[3,:] += -3.*Jma[0,:]

        # Theta 0
        Jma[5,8] += -self.k / (HUB*a_val)
        Jma[5,:] += -Jma[0,:]
        
        # Baryon velocity
        if a_val > tflip_TCA:
            Jma[4,4] += -1. + dTa / (Rfac*HUB*a_val)
            Jma[4,:] += self.k/(HUB*a_val)*PsiTerm
            Jma[4,3] += self.k * CsndB / (HUB * a_val)
            Jma[4,8] += -3.*dTa / (Rfac * HUB * a_val)
        else:
            print 'Use TCA?'
            exit()
            Jma[4,4] += -1./(1.+RR) + 2.*(RR/(1.+RR))**2. + 2.*RR*HUB*a_val/\
                        ((1.+RR)**2.*dTa)
            Jma[4,3] += CsndB*self.k/(HUB*a_val*(1.+RR))
            Jma[4,5] += RR*self.k*(1./(HUB*a_val*(1+RR)) +
                        2./((1.+RR)**2.*dTa))
            Jma[4,11] += -RR*self.k/(2.*HUB*a_val*(1+RR))
            Jma[4,8] += -6.*(RR/(1.+RR))**2.
            Jma[4,:] += (self.k/(HUB*a_val) + RR*self.k /
                        (dTa*(1.+RR)**2.))* PsiTerm
            Jma[4,:] += -(RR*self.k/(dTa*(1.+RR)**2.))*\
                    CsndB*Jma[3,:]
            Jma[4,:] += (RR*self.k/(dTa*(1.+RR)**2.))*Jma[5,:]

        # ThetaP 0
        Jma[6,9] += - self.k / (HUB*a_val)
        Jma[6,6] += dTa / (2.*HUB*a_val)
        Jma[6,11] += - dTa / (2.*HUB*a_val)
        Jma[6,12] += - dTa / (2.*HUB*a_val)

        # Neu 0
        Jma[7,10] += -self.k / (HUB*a_val)
        Jma[7,:] += -Jma[0,:]

        # Theta 1
        if a_val > tflip_TCA:
            Jma[8,5] += self.k/ (3.*HUB*a_val)
            Jma[8,8] += dTa / (HUB*a_val)
            Jma[8,4] += -dTa / (3.*HUB*a_val)
            Jma[8,11] += -2.*self.k / (3.*HUB*a_val)
            Jma[8,:] += self.k*PsiTerm / (3.*HUB*a_val)
        else:
            Jma[8,4] += -1./(3.*RR)
            Jma[8,3] += CsndB*self.k/(HUB*a_val*RR*3.)
            Jma[8,5] += self.k/(3.*HUB*a_val)
            Jma[8,11] += -self.k/(6.*HUB*a_val)
            Jma[8,:] += (1.+RR)*self.k/(3.*RR*HUB*a_val)*PsiTerm
            Jma[8,:] += -Jma[4,:]/(3.*RR)
        
        # ThetaP 1
        Jma[9,6] += self.k / (3.*HUB*a_val)
        Jma[9,12] += -2.*self.k / (3.*HUB*a_val)
        Jma[9,9] += dTa / (HUB*a_val)
        # Neu 1
        Jma[10,7] += self.k / (3.*HUB*a_val)
        Jma[10,13] += -2.*self.k/ (3.*HUB*a_val)
        Jma[10,:] += self.k * PsiTerm / (3.*HUB*a_val)
        # Theta 2
        Jma[11,8] += 2.*self.k / (5.*HUB*a_val)
        Jma[11,14] += -3.*self.k / (5.*HUB*a_val)
        Jma[11,11] += 9.*dTa / (10.*HUB*a_val)
        Jma[11,6] += -dTa / (10.*HUB*a_val)
        Jma[11,12] += -dTa /(10.*HUB*a_val)
        # ThetaP 2
        Jma[12,9] += 2.*self.k / (5.*HUB*a_val)
        Jma[12,15] += -3.*self.k / (5.*HUB*a_val)
        Jma[12,12] += 9.*dTa / (10.*HUB*a_val)
        Jma[12,11] += -dTa / (10.*HUB*a_val)
        Jma[12,6] += -dTa / (10.*HUB*a_val)
        # Neu 2
        Jma[13,10] += 2.*self.k/ (5.*HUB*a_val)
        Jma[13,16] += -3.*self.k/ (5.*HUB*a_val)
        
        for i in range(14, 14 + self.Lmax - 3):
            elV = i - 14 + 3
            inx = i - 14
            Jma[14+3*inx,14+3*inx] += dTa / (HUB*a_val)
            Jma[14+3*inx,14+3*inx-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[14+3*inx,14+3*inx+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            Jma[14+3*inx+1,14+3*inx+1] += dTa / (HUB*a_val)
            Jma[14+3*inx+1,14+3*inx+1-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[14+3*inx+1,14+3*inx+1+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            Jma[14+3*inx+2,14+3*inx+2-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[14+3*inx+2,14+3*inx+2+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

        # Theta Lmax
        Jma[-3, -3-3] += self.k / (HUB*a_val)
        Jma[-3, -3] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val)
        # ThetaP Lmax
        Jma[-2, -2-3] += self.k / (HUB*a_val)
        Jma[-2, -2] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val)
        # Nu Lmax
        Jma[-1, -1-3] += self.k / (HUB*a_val)
        Jma[-1, -1] += -(self.Lmax+1.)/(eta*HUB*a_val)
        
        return Jma

    def scale_a(self, eta):
        return 10.**self.ct_to_scale(np.log10(eta))
    
    def conform_T(self, a):
        return quad(lambda x: 1./self.H_0 /np.sqrt(self.omega_R+self.omega_M*x+self.omega_L*x**4.), 0., a)[0]

    def hubble(self, a):
        return self.H_0*np.sqrt(self.omega_R*a**-4.+self.omega_M*a**-3.+self.omega_L)

    def rhoCDM(self, a):
        return self.omega_cdm * self.H_0**2. * a**-3.

    def rhoB(self, a):
        return self.omega_b * self.H_0**2. * a**-3.

    def rhoG(self, a):
        return self.omega_g * self.H_0**2. * a**-4.

    def rhoNeu(self, a):
        return self.omega_nu * self.H_0**2. * a**-4.

    def epsilon_test(self, a):
        denom = (self.omega_M*a**-3. + self.omega_R*a**-4. + self.omega_L)
        phiTerm = -2./3.*(self.k/(a*self.H_0))**2.*self.combined_vector[0][-1]
        denTerm = (self.omega_cdm*self.combined_vector[1][-1]+self.omega_b*self.combined_vector[3][-1])*a**-3. +\
                  4.*(self.omega_g*self.combined_vector[5][-1]+self.omega_nu*self.combined_vector[7][-1])*a**-4.
        velTerm = 3.*a*self.hubble(a)/self.k*(
                 (self.omega_cdm*self.combined_vector[2][-1]+self.omega_b*self.combined_vector[4][-1])*a**-3. +
                 4.*(self.omega_g*self.combined_vector[8][-1]+self.omega_nu*self.combined_vector[10][-1])*a**-4.)
        return (phiTerm + denTerm + velTerm)/denom


    def save_system(self):
        psi_term = np.zeros(len(self.eta_vector))
        for i in range(len(self.eta_vector)):
            aval = 10.**self.ct_to_scale(np.log10(self.eta_vector[i]))
            psi_term[i] = -12.*(aval**2./self.k**2.*(self.rhoNeu(aval)*self.combined_vector[13][i] +
                                                       self.rhoG(aval)*self.combined_vector[11][i])) - self.combined_vector[0][i]
        
        sve_tab = np.zeros((len(self.eta_vector), self.TotalVars+2))
        sve_tab[:,0] = self.eta_vector
        sve_tab[:,-1] = psi_term
        for i in range(self.TotalVars):
            sve_tab[:,i+1] = self.combined_vector[i]
        np.savetxt(path + '/OutputFiles/StandardUniverse_FieldEvolution_{:.4e}.dat'.format(self.k), sve_tab, fmt='%.8e', delimiter='    ')
        
        if self.testing:
            np.savetxt(path+'/OutputFiles/StandardUniverse_Background.dat',
                        np.column_stack((self.aLIST, self.etaLIST, self.xeLIST, self.hubLIST, self.csLIST, self.dtauLIST)))
        return

class ManyBrane_Universe(object):
    
    def __init__(self, Nbrane, k, omega_b, omega_cdm, omega_g, omega_L, omega_nu, accuracy=1e-3,
                 stepsize=0.01, lmax=5, testing=False):
        self.omega_b_T = omega_b[0] + Nbrane*omega_b[1]
        self.omega_cdm_T = omega_cdm[0] + Nbrane*omega_cdm[1]
        self.omega_g_T = omega_g[0] + Nbrane*omega_g[1]
        self.omega_nu_T = omega_nu[0] + Nbrane*omega_nu[1]
        self.omega_M_T = self.omega_b_T + self.omega_cdm_T
        self.omega_R_T = self.omega_nu_T + self.omega_g_T
        self.omega_L_T = np.sum(1. - self.omega_M_T - self.omega_R_T)
        self.omega_b = omega_b
        self.omega_cdm = omega_cdm
        self.omega_g = omega_g
        self.omega_nu = omega_nu
        self.Nbrane = Nbrane
        
        self.darkCMB_T = 2.7255 * (omega_g[1] / omega_g[0])**0.25
        
        self.PressureFac = (self.omega_g[1] / self.omega_b[1]) / (self.omega_g[0] / self.omega_b[0])
        self.ECDM = self.omega_cdm_T
        
        ngamma_pr = 410.7 * (self.darkCMB_T/2.7255)**3.
        nbarys = 2.503e-7 * (omega_b[1]/omega_b[0])
        etaPr = 6.1e-10 * (omega_b[1]/omega_b[0])*(omega_g[0]/omega_g[1])
        self.yp_prime = Yp_Prime(etaPr)
        
        print 'Fraction of baryons on each brane: {:.3f}'.format(omega_b[1]/omega_b[0])
        
        self.H_0 = 2.2348e-4 # units Mpc^-1
        #self.eta_0 = 1.4387e4

        self.Lmax = lmax
        self.stepsize = stepsize
        
        self.k = k
        print 'Solving perturbations for k = {:.3e} \n'.format(k)
        
        self.accuracy = accuracy
        self.TotalVars = 8 + 3*self.Lmax
        self.step = 0
        
        self.Theta_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Theta_P_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        self.Neu_Dot = np.zeros(self.Lmax+1 ,dtype=object)
        
        self.Theta_Dot_D = np.zeros(self.Lmax+1 ,dtype=object)
        self.Theta_P_Dot_D = np.zeros(self.Lmax+1 ,dtype=object)
        self.Neu_Dot_D = np.zeros(self.Lmax+1 ,dtype=object)
        
        self.combined_vector = np.zeros(2*self.TotalVars-1 ,dtype=object)
        self.Psi_vec = []
        self.Phi_vec = []
        self.combined_vector[0] = self.Phi_vec
        self.combined_vector[1] = self.dot_rhoCDM_vec = []
        self.combined_vector[2] = self.dot_velCDM_vec = []
        self.combined_vector[3] = self.dot_rhoB_vec = []
        self.combined_vector[4] = self.dot_velB_vec = []
        for i in range(self.Lmax + 1):
            self.combined_vector[5+i*3] = self.Theta_Dot[i] = []
            self.combined_vector[6+i*3] = self.Theta_P_Dot[i] = []
            self.combined_vector[7+i*3] = self.Neu_Dot[i] = []
        
        self.combined_vector[self.TotalVars] = self.dot_rhoCDM_vec_D = []
        self.combined_vector[self.TotalVars+1] = self.dot_velCDM_vec_D = []
        self.combined_vector[self.TotalVars+2] = self.dot_rhoB_vec_D = []
        self.combined_vector[self.TotalVars+3] = self.dot_velB_vec_D = []
        for i in range(self.Lmax + 1):
            self.combined_vector[self.TotalVars+4+i*3] = self.Theta_Dot_D[i] = []
            self.combined_vector[self.TotalVars+5+i*3] = self.Theta_P_Dot_D[i] = []
            self.combined_vector[self.TotalVars+6+i*3] = self.Neu_Dot_D[i] = []
        
#        self.load_funcs()
        self.compute_funcs()
        
        self.testing = testing
        if self.testing:
            self.aLIST = []
            self.etaLIST = []
            self.csLIST = []
            self.hubLIST = []
            self.dtauLIST = []
            self.xeLIST = []
            self.csD_LIST = []
            self.dtauD_LIST = []
            self.xeD_LIST = []

        return

    def compute_funcs(self):
        a0_init = np.logspace(-14, 0, 1e4)
        eta_list = np.zeros_like(a0_init)
        for i in range(len(a0_init)):
            eta_list[i] = self.conform_T(a0_init[i])
        self.eta_0 = eta_list[-1]
        self.ct_to_scale = interp1d(np.log10(eta_list), np.log10(a0_init), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(a0_init), np.log10(eta_list), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')

        self.Thermal_sln()
        return

    def clearfiles(self):
        if os.path.isfile(path + '/precomputed/xe_working.dat'):
            os.remove(path + '/precomputed/xe_working.dat')
        if os.path.isfile(path + '/precomputed/tb_working.dat'):
            os.remove(path + '/precomputed/tb_working.dat')
        
        if os.path.isfile(path + '/precomputed/xe_dark_working.dat'):
            os.remove(path + '/precomputed/xe_dark_working.dat')
        if os.path.isfile(path + '/precomputed/tb_dark_working.dat'):
            os.remove(path + '/precomputed/tb_dark_working.dat')

        if os.path.isfile(path + '/precomputed/working_expOpticalDepth.dat'):
            os.remove(path + '/precomputed/working_expOpticalDepth.dat')
        if os.path.isfile(path + '/precomputed/working_VisibilityFunc.dat'):
            os.remove(path + '/precomputed/working_VisibilityFunc.dat')
        return
    
    def Thermal_sln(self):
        self.tb_fileNme = path + '/precomputed/tb_working.dat'
        self.Xe_fileNme = path + '/precomputed/xe_working.dat'
        
        self.tbDk_fileNme = path + '/precomputed/tb_dark_working.dat'
        self.Xedk_fileNme = path + '/precomputed/xe_dark_working.dat'
        
        if not os.path.isfile(self.tb_fileNme) or not os.path.isfile(self.Xe_fileNme):
            tvals = np.linspace(3.5, -1, 1000)
            y0 = [1., 2.7255 * (1. + 10.**tvals[0]), 1., self.darkCMB_T * (1. + 10.**tvals[0])]
         
            val_sln = odeint(self.thermal_funcs, y0, tvals)
            avals = 1. / (1. + 10.**tvals)
            val_sln[:,0][10.**tvals <= 7.68] = 1.
            self.Tb_1 = np.column_stack((avals, val_sln[:, 1]))
            np.savetxt(self.tb_fileNme, self.Tb_1)
            self.Xe_1 = np.column_stack((avals, val_sln[:,0]))
            np.savetxt(self.Xe_fileNme, self.Xe_1)
            self.Xe_dark = np.column_stack((avals, val_sln[:,2]))
            np.savetxt(self.Xedk_fileNme, self.Xe_dark)
            self.Tb_drk = np.column_stack((avals, val_sln[:,3]))
            np.savetxt(self.tbDk_fileNme, self.Tb_drk)
        else:
            self.Tb_1 = np.loadtxt(self.tb_fileNme)
            self.Xe_1 = np.loadtxt(self.Xe_fileNme)
            self.Tb_drk = np.loadtxt(self.tbDk_fileNme)
            self.Xe_dark = np.loadtxt(self.Xedk_fileNme)
        
        self.Tb = interp1d(np.log10(self.Tb_1[:,0]), np.log10(self.Tb_1[:,1]), bounds_error=False, fill_value='extrapolate')
        self.Xe = interp1d(np.log10(self.Xe_1[:,0]), np.log10(self.Xe_1[:,1]), bounds_error=False, fill_value='extrapolate')
        self.Tb_D = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), bounds_error=False, fill_value='extrapolate')
        self.XE_DARK_B = interp1d(np.log10(self.Xe_dark[:,0]), np.log10(self.Xe_dark[:,1]), bounds_error=False, fill_value=0.)
        return

    def Tb_DARK(self, a):
        
        if a < 1:
            z = (1./a - 1.)
        else:
            z = 0.1
        
        if z <= 10.**3.5:
            return 10.**self.Tb_D(np.log10(a))
        else:
            return (z - (1./self.Tb_drk[0,0] - 1.))*self.Tb_drk[0,1] + self.Tb_drk[0,1]

    def thermal_funcs(self, val, z):
        xe, T, xeD, TD = val
        return [self.xeDiff([xe], z, T)[0], self.dotT([T], z, xe), self.xeDiff([xeD], z, TD, dark=True)[0], self.dotT([TD], z, xeD, dark=True)]

    def dotT(self, T, lgz, xe, dark=False):
        kb = 8.617e-5/1e9 # Gev/K
        thompson_xsec = 6.65e-25 # cm^2
        aval = 1. / (1. + 10.**lgz)
        if not dark:
            Yp = 0.245
        else:
            Yp = self.yp_prime
        
        Mpc_to_cm = 3.086e24
        
        if xe >= 1.:
            mol_wei = 0.5*(1.-Yp) + Yp*1.33
        else:
            mol_wei = 1.*(1.-Yp) + Yp*4.
        
        n_b = 2.503e-7*(1.+10.**lgz)**3.
        if dark:
            n_b *= self.omega_b[1]/self.omega_b[0]
        hub = self.hubble(aval)
        if dark:
            omega_Rat = self.omega_g[1] / self.omega_b[1]
        else:
            omega_Rat = self.omega_g[0] / self.omega_b[0]
        jacF = - 1. * (10.**lgz * np.log(10.))
        return (-2.*T[0]*aval + (1./hub)*(8./3.)*(mol_wei/5.11e-4)*omega_Rat*(xe*n_b*thompson_xsec)*(2.7255*(1.+10.**lgz) - T[0])*Mpc_to_cm)*jacF
    
    def Cs_Sqr(self, a, dark=False):
        kb = 8.617e-5/1e9 # GeV/K
        
        if not dark:
            facxe = 10.**self.Xe(np.log10(a))
            Yp = 0.245
            Tb = 10.**self.Tb(np.log10(a))
        else:
            facxe = 10.**self.XE_DARK_B(np.log10(a))
            Yp = self.yp_prime
            Tb = self.Tb_DARK(a)
    
        mol_wei = np.zeros_like(facxe)
        mol_wei[facxe >= 1] = 0.5*(1.-Yp) + Yp*1.33
        mol_wei[facxe < 1] = 1.*(1.-Yp) + Yp*4.
        
        if a < 1:
            lgZ = np.log10(1./a - 1.)
        else:
            lgZ = -10
        
        extraPT = self.dotT([Tb], lgZ, facxe, dark=dark)*(-1./Tb)*(1.+10.**lgZ)/(np.log(10.) * 10.**lgZ)
        val_r = kb*Tb/mol_wei*(1. - 1./3. * extraPT)
        if val_r < 0.:
            return 0.
        return val_r
    
    def xeDiff(self, val, y, tgas, dark=False, hydrogen=True, first=True):
        yy = 10.**y
     
        if hydrogen:
            ep0 = 13.6/1e9  # GeV
        else:
            if first:
                ep0 = 54.4/1e9
            else:
                ep0 = 24.6/1e9

        kb = 8.617e-5/1e9 # Gev/K
        GeV_cm = 5.06e13
        Mpc_to_cm = 3.086e24
        me = 5.11e-4 # GeV
        aval = 1. / (1.+yy)
        if not dark:
            Yp = 0.245
        else:
            Yp = self.yp_prime
        n_b = 2.503e-7 / aval**3.
        if dark:
            n_b *= self.omega_b[1]/self.omega_b[0]
        hub = self.hubble(aval)
        FScsnt = 7.29e-3
        
        alpha2 = 9.78*(FScsnt/me)**2.*np.sqrt(ep0/(kb*tgas))*np.log(ep0/(kb*tgas))/(GeV_cm**2.) # cm^2
        beta = alpha2*(me*(kb*tgas)/(2.*np.pi))**(3./2.)*np.exp(-ep0/(kb*tgas))*GeV_cm**3. # 1/cm
        beta2 = alpha2*(me*(kb*tgas)/(2.*np.pi))**(3./2.)*np.exp(-ep0/(4.*kb*tgas))*GeV_cm**3.*Mpc_to_cm
        
        if val[0] > 0.999:
            Cr = 1.
        else:
            Lalpha = (3.*ep0)**3.*hub / (64.*np.pi**2*(1-val[0])*n_b*Yp) * GeV_cm**3.
            L2g = 8.227 / 2.998e10 * Mpc_to_cm
            Cr = (Lalpha + L2g) / (Lalpha + L2g + beta2)
        
        Value = Cr*np.log(10.)*yy*(-aval)/(hub)*((1.-val[0])*beta - val[0]**2.*n_b*alpha2)*Mpc_to_cm
        return [Value]
    
    
    def tau_functions(self):
        self.fileN_optdep = path + '/precomputed/working_expOpticalDepth.dat'
        self.fileN_visibil = path + '/precomputed/working_VisibilityFunc.dat'
        Mpc_to_cm = 3.086e24
        if not os.path.isfile(self.fileN_visibil) or not os.path.isfile(self.fileN_optdep):
            avals = np.logspace(-7, 0, 1000)
            Yp = 0.245
            n_b = 2.503e-7 / avals**3.
            thompson_xsec = 6.65e-25 # cm^2
            xevals = 10.**self.Xe(np.log10(avals))
            hubbs = self.hubble(avals)
            dtau = -xevals * (1. - Yp) * n_b * thompson_xsec * avals * Mpc_to_cm
            dtau_I = interp1d(10.**self.scale_to_ct(np.log10(avals)), dtau, kind='linear', bounds_error=False, fill_value='extrapolate')
            tau = np.zeros_like(dtau)
            for i in range(len(dtau)):
                tau[i] = -np.trapz(dtau[i:], 10.**self.scale_to_ct(np.log10(avals[i:])))
            tau[0] = tau[1]
            np.savetxt(self.fileN_optdep, np.column_stack((avals, np.exp(-tau))))
            np.savetxt(self.fileN_visibil, np.column_stack((avals, -dtau * np.exp(-tau))))
    
        return

    def init_conds(self, eta_0, aval):
        OM = self.omega_M_T * self.H_0**2./self.hubble(aval)**2./aval**3.
        OR = self.omega_R_T * self.H_0**2./self.hubble(aval)**2./aval**4.
        ONu = self.omega_nu_T * self.H_0**2./self.hubble(aval)**2./aval**4.
        rfactor = ONu / (0.75*OM*aval + OR)

        self.inital_perturb = -1./6.
        for i in range(1):
            self.Psi_vec.append(self.inital_perturb)
            self.Phi_vec.append(-(1.+2.*rfactor/5.)*self.Psi_vec[-1])
            
            self.dot_rhoCDM_vec.append(-3./2.*self.Psi_vec[-1])
            self.dot_velCDM_vec.append(1./2*eta_0*self.k*self.Psi_vec[-1])
            self.dot_rhoB_vec.append(-3./2.*self.Psi_vec[-1])
            self.dot_velB_vec.append(1./2*eta_0*self.k*self.Psi_vec[-1])
            self.Theta_Dot[0].append(-1./2.*self.Psi_vec[-1])
            self.Theta_Dot[1].append(1./6*eta_0*self.k*self.Psi_vec[-1])
            self.Neu_Dot[0].append(-1./2.*self.Psi_vec[-1])
            self.Neu_Dot[1].append(1./6*eta_0*self.k*self.Psi_vec[-1])
            self.Neu_Dot[2].append(1./30.*(self.k*eta_0)**2.*self.Psi_vec[-1])
            
            self.dot_rhoCDM_vec_D.append(-3./2.*self.Psi_vec[-1])
            self.dot_velCDM_vec_D.append(1./2*eta_0*self.k*self.Psi_vec[-1])
            self.dot_rhoB_vec_D.append(-3./2.*self.Psi_vec[-1])
            self.dot_velB_vec_D.append(1./2*eta_0*self.k*self.Psi_vec[-1])
            self.Theta_Dot_D[0].append(-1./2.*self.Psi_vec[-1])
            self.Theta_Dot_D[1].append(1./6*eta_0*self.k*self.Psi_vec[-1])
            self.Neu_Dot_D[0].append(-1./2.*self.Psi_vec[-1])
            self.Neu_Dot_D[1].append(1./6*eta_0*self.k*self.Psi_vec[-1])
            self.Neu_Dot_D[2].append(1./30.*(self.k*eta_0)**2.*self.Psi_vec[-1])
            
            for i in range(self.Lmax + 1):
                if i > 1:
                    self.Theta_Dot[i].append(0.)
                    self.Theta_Dot_D[i].append(0.)
                self.Theta_P_Dot[i].append(0.)
                self.Theta_P_Dot_D[i].append(0.)
                if i > 2:
                    self.Neu_Dot[i].append(0.)
                    self.Neu_Dot_D[i].append(0.)

#        for i in range(1,self.TotalVars):
#            print self.combined_vector[i], self.combined_vector[self.TotalVars+i-1]

        self.step = 0
        return
    
    
    def solve_system(self):
        eta_st = np.min([1e-3/self.k, 1e-1/0.7]) # Initial conformal time in Mpc
        y_st = np.log(self.scale_a(eta_st))
        eta_st = self.conform_T(np.exp(y_st))
        
        self.init_conds(eta_st, np.exp(y_st))
        self.eta_vector = [eta_st]
        self.y_vector = [y_st]
        
#        test initial conditions
#        self.epsilon_test(np.exp(self.y_vector[-1]))
#        exit()

        try_count = 0.
        try_max = 20.
        FailRUN = False
        last_step_up = False
        while (self.eta_vector[-1] < (self.eta_0-1.)):
            if try_count > try_max:
                #print 'FAIL TRY MAX....Breaking.'
                FailRUN=True
                break
            y_use = self.y_vector[-1] + self.stepsize
            eta_use = self.conform_T(np.exp(y_use))
            if (eta_use > self.eta_0):
                eta_use = self.eta_0
                y_use = np.log(self.scale_a(eta_use))
            self.eta_vector.append(eta_use)

            y_diff = y_use - self.y_vector[-1]
            self.y_vector.append(y_use)
            
            if self.step%3000 == 0:
                print 'Last a: {:.7e}, New a: {:.7e}'.format(np.exp(self.y_vector[-2]), np.exp(self.y_vector[-1]))
            if ((y_diff > eta_use*np.exp(y_use)*self.hubble(np.exp(y_use))) or
                (y_diff > np.max([np.exp(y_use)*self.hubble(np.exp(y_use)),
                                  np.exp(y_use)*self.hubble(np.exp(y_use))/self.k]))):
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                #print 'Failed test of expansion/Courant time...'
                try_count += 1
                continue
            self.step_solver()
    
            test_epsilon = self.epsilon_test(np.exp(self.y_vector[-1]))
#            print test_epsilon
#            if np.abs(test_epsilon) > 10 and self.step > 10:
#                raise ValueError
#                continue
            self.step += 1
            if (np.abs(test_epsilon) < 1e-4*self.accuracy) and not last_step_up:
                self.stepsize *= 1.25
                last_step_up = True
                #print 'Increase Step Size'
            else:
                last_step_up = False
            try_count = 0.
        
        if not FailRUN:
            print 'Saving File...'
            self.save_system()
        return

    def step_solver(self):
        if self.step > 0:
            tau_n = (self.y_vector[-1] - self.y_vector[-2]) / (self.y_vector[-2] - self.y_vector[-3])
        else:
            tau_n = (self.y_vector[-1] - self.y_vector[-2]) / self.y_vector[-2]

        delt = (self.y_vector[-1] - self.y_vector[-2])
        Ident = np.eye(self.TotalVars*2 - 1)
        Jmat = self.matrix_J(self.y_vector[-1])
        
        Amat = (1.+2.*tau_n)/(1.+tau_n)*Ident - delt*Jmat
        bvec = self.b_vector(tau_n)
        ysol = np.matmul(inv(Amat),bvec)
        for i in range(2*self.TotalVars - 1):
            self.combined_vector[i].append(ysol[i])
#        for i in range(1, self.TotalVars):
#            print self.combined_vector[i][-1],self.combined_vector[self.TotalVars+i-1][-1]

        return
    
    def b_vector(self, tau):
        bvec = np.zeros(2*self.TotalVars - 1)
        for i in range(2*self.TotalVars - 1):
            if self.step == 0:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1]
            else:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1] - tau**2./(1.+tau)*self.combined_vector[i][-2]
        return bvec
    
    def matrix_J(self, z_val):
        a_val = np.exp(z_val)
        eta = self.conform_T(a_val)
        HUB = self.hubble(a_val)
        
        Jma = np.zeros((2*self.TotalVars-1, 2*self.TotalVars-1))
        
        Rfac = (3.*self.omega_b[0]*a_val)/(4.*self.omega_g[0])
        RR = (4.*self.omega_g[0])/(3.*self.omega_b[0]*a_val)
        #Rfac_D = (3.*self.omega_b[1]*a_val)/(4.*self.omega_g[1])
        RR_D = (4.*self.omega_g[1])/(3.*self.omega_b[1]*a_val)
        
        Yp = 0.245
        n_b = 2.503e-7
        dTa = -self.xe_deta(a_val)*(1.-Yp)*n_b*6.65e-29*1e4/a_val**2./3.24078e-25
       
        xeDk = 10.**self.XE_DARK_B(np.log10(a_val))
        dTa_D = -xeDk*(1.-self.yp_prime)*n_b*6.65e-29*1e4/ a_val**2./3.24078e-25*(self.omega_b[1]/self.omega_b[0])
        
        CsndB = self.Cs_Sqr(a_val, dark=False)
        CsndB_D = self.Cs_Sqr(a_val, dark=True)
        
        
        if self.testing:
            self.aLIST.append(a_val)
            self.etaLIST.append(eta)
            self.hubLIST.append(HUB)
            self.csLIST.append(CsndB)
            self.dtauLIST.append(dTa)
            self.xeLIST.append(self.xe_deta(a_val))
            self.csD_LIST.append(CsndB_D)
            self.dtauD_LIST.append(dTa_D)
            self.xeD_LIST.append(xeDk)
        
        #tflip_TCA = 1e-4
        tflip_TCA = 1e-11
        
        PsiTerm = np.zeros(2*self.TotalVars-1)
        PsiTerm[0] = -1.
        PsiTerm[11] = -12.*(a_val/self.k)**2.*self.omega_g[0]*self.H_0**2./a_val**4.
        PsiTerm[13] = -12.*(a_val/self.k)**2.*self.omega_nu[0]*self.H_0**2./a_val**4.
        PsiTerm[self.TotalVars + 11 - 1] = -12.*(a_val/self.k)**2.*self.omega_g[1]*self.H_0**2./a_val**4.*self.Nbrane
        PsiTerm[self.TotalVars + 13 - 1] = -12.*(a_val/self.k)**2.*self.omega_nu[1]*self.H_0**2./a_val**4.*self.Nbrane
        
        # Phi Time derivative
        Jma[0,:] += PsiTerm
        Jma[0,0] += -((self.k/(HUB*a_val))**2.)/3.
        
        Jma[0,1] += 1./(HUB**2.*2.)*self.rhoCDM_Indiv(a_val, uni=0)
        Jma[0,3] += 1./(HUB**2.*2.)*self.rhoB_Indiv(a_val, uni=0)
        Jma[0,5] += 2./(HUB**2.)*self.rhoG_Indiv(a_val, uni=0)
        Jma[0,7] += 2./(HUB**2.)*self.rhoNeu_Indiv(a_val, uni=0)
        
        Jma[0, self.TotalVars] += 1./(HUB**2.*2.)*self.rhoCDM_Indiv(a_val, uni=1) * self.Nbrane
        Jma[0, self.TotalVars + 2] += 1./(HUB**2.*2.)*self.rhoB_Indiv(a_val, uni=1) * self.Nbrane
        Jma[0, self.TotalVars + 4] += 2./(HUB**2.)*self.rhoG_Indiv(a_val, uni=1) * self.Nbrane
        Jma[0, self.TotalVars + 6] += 2./(HUB**2.)*self.rhoNeu_Indiv(a_val, uni=1) * self.Nbrane

        # CDM density
        Jma[1,2] += -self.k/(HUB*a_val)
        Jma[1,:] += -3.*Jma[0,:]
        
        Jma[self.TotalVars, self.TotalVars + 1] += -self.k/(HUB*a_val)
        Jma[self.TotalVars,:] += -3.*Jma[0,:]

        # CDM velocity
        Jma[2,2] += -1.
        Jma[2,:] += self.k/(HUB*a_val)*PsiTerm
        
        Jma[self.TotalVars+1,self.TotalVars+1] += -1.
        Jma[self.TotalVars+1,:] += self.k/(HUB*a_val)*PsiTerm

        # Baryon density
        Jma[3,4] += -self.k / (HUB*a_val)
        Jma[3,:] += -3.*Jma[0,:]
        
        Jma[self.TotalVars+2,self.TotalVars+3] += -self.k / (HUB*a_val)
        Jma[self.TotalVars+2,:] += -3.*Jma[0,:]

        # Theta 0
        Jma[5,8] += -self.k / (HUB*a_val)
        Jma[5,:] += -Jma[0,:]
        
        Jma[self.TotalVars+4,self.TotalVars+7] += -self.k / (HUB*a_val)
        Jma[self.TotalVars+4,:] += -Jma[0,:]
        
        # Baryon velocity
        if a_val > tflip_TCA:
            Jma[4,4] += -1. + dTa / (Rfac*HUB*a_val)
            Jma[4,:] += self.k/(HUB*a_val)*PsiTerm
            Jma[4,3] += self.k * CsndB / (HUB * a_val)
            Jma[4,8] += -3.*dTa / (Rfac * HUB * a_val)
        
            Jma[self.TotalVars+3,self.TotalVars+3] += -1. + dTa_D / (HUB*a_val) * RR_D
            Jma[self.TotalVars+3,:] += self.k/(HUB*a_val)*PsiTerm
            Jma[self.TotalVars+3,self.TotalVars+2] += self.k * CsndB_D / (HUB * a_val)
            Jma[self.TotalVars+3,self.TotalVars+7] += -3.*dTa_D / (HUB * a_val) * RR_D
        
        else:
            Jma[4,4] += -1./(1.+RR) + 2.*(RR/(1.+RR))**2. + 2.*RR*HUB*a_val/\
                        ((1.+RR)**2.*dTa)
            Jma[4,3] += CsndB*self.k/(HUB*a_val*(1.+RR))
            Jma[4,5] += RR*self.k*(1./(HUB*a_val*(1+RR)) +
                        2./((1.+RR)**2.*dTa))
            Jma[4,11] += -RR*self.k/(2.*HUB*a_val*(1+RR))
            Jma[4,8] += -6.*(RR/(1.+RR))**2.
            Jma[4,:] += (self.k/(HUB*a_val) + RR*self.k /
                        (dTa*(1.+RR)**2.))* PsiTerm
            Jma[4,:] += -(RR*self.k/(dTa*(1.+RR)**2.))*\
                    CsndB*Jma[3,:]
            Jma[4,:] += (RR*self.k/(dTa*(1.+RR)**2.))*Jma[5,:]
            
            Jma[self.TotalVars+3,self.TotalVars+3] += -1./(1.+RR_D)+2.*(RR_D/(1.+RR_D))**2.+ \
                                                    2.*RR_D*HUB*a_val/((1.+RR_D)**2.*dTa_D)
            Jma[self.TotalVars+3,self.TotalVars+2] += CsndB_D*self.k/(HUB*a_val*(1.+RR_D))
            Jma[self.TotalVars+3,self.TotalVars+4] += RR_D*self.k*(1./(HUB*a_val*(1+RR_D)) + \
                                                      2./((1.+RR_D)**2.*dTa_D))
            Jma[self.TotalVars+3,self.TotalVars+10] += -RR_D*self.k/(2.*HUB*a_val*(1+RR_D))
            Jma[self.TotalVars+3,self.TotalVars+7] += -6.*(RR_D/(1.+RR_D))**2.
            Jma[self.TotalVars+3,:] += (self.k/(HUB*a_val) + RR_D*self.k /
                                        (dTa_D*(1.+RR_D)**2.))* PsiTerm
            Jma[self.TotalVars+3,:] += -(RR_D*self.k/(dTa_D*(1.+RR_D)**2.))* \
                                        CsndB_D*Jma[self.TotalVars+2,:]
            Jma[self.TotalVars+3,:] += (RR_D*self.k/(dTa_D*(1.+RR_D)**2.))* \
                                        Jma[self.TotalVars+4,:]

        # ThetaP 0
        Jma[6,9] += - self.k / (HUB*a_val)
        Jma[6,6] += dTa / (2.*HUB*a_val)
        Jma[6,11] += - dTa / (2.*HUB*a_val)
        Jma[6,12] += - dTa / (2.*HUB*a_val)
        
        Jma[self.TotalVars+5,self.TotalVars+8] += - self.k / (HUB*a_val)
        Jma[self.TotalVars+5,self.TotalVars+5] += dTa_D / (2.*HUB*a_val)
        Jma[self.TotalVars+5,self.TotalVars+10] += - dTa_D / (2.*HUB*a_val)
        Jma[self.TotalVars+5,self.TotalVars+11] += - dTa_D / (2.*HUB*a_val)

        # Neu 0
        Jma[7,10] += -self.k / (HUB*a_val)
        Jma[7,:] += -Jma[0,:]
        
        Jma[self.TotalVars+6,self.TotalVars+9] += -self.k / (HUB*a_val)
        Jma[self.TotalVars+6,:] += -Jma[0,:]

        # Theta 1
        if a_val > tflip_TCA:
            Jma[8,5] += self.k/ (3.*HUB*a_val)
            Jma[8,8] += dTa / (HUB*a_val)
            Jma[8,4] += -dTa / (3.*HUB*a_val)
            Jma[8,11] += -2.*self.k / (3.*HUB*a_val)
            Jma[8,:] += self.k*PsiTerm / (3.*HUB*a_val)
        
            Jma[self.TotalVars+7,self.TotalVars+4] += self.k/ (3.*HUB*a_val)
            Jma[self.TotalVars+7,self.TotalVars+7] += dTa_D / (HUB*a_val)
            Jma[self.TotalVars+7,self.TotalVars+3] += -dTa_D / (3.*HUB*a_val)
            Jma[self.TotalVars+7,self.TotalVars+10] += -2.*self.k / (3.*HUB*a_val)
            Jma[self.TotalVars+7,:] += self.k*PsiTerm / (3.*HUB*a_val)
        else:
            Jma[8,4] += -1./(3.*RR)
            Jma[8,3] += CsndB*self.k/(HUB*a_val*RR*3.)
            Jma[8,5] += self.k/(3.*HUB*a_val)
            Jma[8,11] += -self.k/(6.*HUB*a_val)
            Jma[8,:] += (1.+RR)*self.k/(3.*RR*HUB*a_val)*PsiTerm
            Jma[8,:] += -Jma[4,:]/(3.*RR)

            Jma[self.TotalVars+7,self.TotalVars+3] += -1./(3.*RR_D)
            Jma[self.TotalVars+7,self.TotalVars+2] += CsndB_D*self.k/(HUB*a_val*RR_D*3.)
            Jma[self.TotalVars+7,self.TotalVars+4] += self.k/(3.*HUB*a_val)
            Jma[self.TotalVars+7,self.TotalVars+10] += -self.k/(6.*HUB*a_val)
            Jma[self.TotalVars+7,:] += (1.+RR_D)*self.k/(3.*RR_D*HUB*a_val)*PsiTerm
            Jma[self.TotalVars+7,:] += -Jma[self.TotalVars+3,:]/(3.*RR_D)
        
        # ThetaP 1
        Jma[9,6] += self.k / (3.*HUB*a_val)
        Jma[9,12] += -2.*self.k / (3.*HUB*a_val)
        Jma[9,9] += dTa / (HUB*a_val)

        Jma[self.TotalVars+8,self.TotalVars+5] += self.k / (3.*HUB*a_val)
        Jma[self.TotalVars+8,self.TotalVars+11] += -2.*self.k / (3.*HUB*a_val)
        Jma[self.TotalVars+8,self.TotalVars+8] += dTa_D / (HUB*a_val)

        # Neu 1
        Jma[10,7] += self.k / (3.*HUB*a_val)
        Jma[10,13] += -2.*self.k/ (3.*HUB*a_val)
        Jma[10,:] += self.k * PsiTerm / (3.*HUB*a_val)

        Jma[self.TotalVars+9,self.TotalVars+6] += self.k / (3.*HUB*a_val)
        Jma[self.TotalVars+9,self.TotalVars+12] += -2.*self.k/ (3.*HUB*a_val)
        Jma[self.TotalVars+9,:] += self.k * PsiTerm / (3.*HUB*a_val)

        # Theta 2
        Jma[11,8] += 2.*self.k / (5.*HUB*a_val)
        Jma[11,14] += -3.*self.k / (5.*HUB*a_val)
        Jma[11,11] += 9.*dTa / (10.*HUB*a_val)
        Jma[11,6] += - dTa / (10.*HUB*a_val)
        Jma[11,12] += - dTa /(10.*HUB*a_val)

        Jma[self.TotalVars+10,self.TotalVars+7] += 2.*self.k / (5.*HUB*a_val)
        Jma[self.TotalVars+10,self.TotalVars+13] += -3.*self.k / (5.*HUB*a_val)
        Jma[self.TotalVars+10,self.TotalVars+10] += 9.*dTa_D / (10.*HUB*a_val)
        Jma[self.TotalVars+10,self.TotalVars+5] += - dTa_D / (10.*HUB*a_val)
        Jma[self.TotalVars+10,self.TotalVars+11] += - dTa_D /(10.*HUB*a_val)

        # ThetaP 2
        Jma[12,9] += 2.*self.k / (5.*HUB*a_val)
        Jma[12,15] += -3.*self.k / (5.*HUB*a_val)
        Jma[12,12] += 9.*dTa / (10.*HUB*a_val)
        Jma[12,11] += -dTa / (10.*HUB*a_val)
        Jma[12,6] += - dTa / (10.*HUB*a_val)

        Jma[self.TotalVars+11,self.TotalVars+8] += 2.*self.k / (5.*HUB*a_val)
        Jma[self.TotalVars+11,self.TotalVars+14] += -3.*self.k / (5.*HUB*a_val)
        Jma[self.TotalVars+11,self.TotalVars+11] += 9.*dTa_D / (10.*HUB*a_val)
        Jma[self.TotalVars+11,self.TotalVars+10] += -dTa_D / (10.*HUB*a_val)
        Jma[self.TotalVars+11,self.TotalVars+5] += - dTa_D / (10.*HUB*a_val)

        # Neu 2
        Jma[13,10] += 2.*self.k/ (5.*HUB*a_val)
        Jma[13,16] += -3.*self.k/ (5.*HUB*a_val)

        Jma[self.TotalVars+12,self.TotalVars+9] += 2.*self.k/ (5.*HUB*a_val)
        Jma[self.TotalVars+12,self.TotalVars+15] += -3.*self.k/ (5.*HUB*a_val)
        
        for i in range(14, 14 + self.Lmax - 3):
            elV = i - 14 + 3
            inx = i - 14
            Jma[14+3*inx,14+3*inx] += dTa / (HUB*a_val)
            Jma[14+3*inx,14+3*inx-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[14+3*inx,14+3*inx+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            Jma[14+3*inx+1,14+3*inx+1] += dTa / (HUB*a_val)
            Jma[14+3*inx+1,14+3*inx+1-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[14+3*inx+1,14+3*inx+1+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            Jma[14+3*inx+2,14+3*inx+2-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[14+3*inx+2,14+3*inx+2+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            inx_D = self.TotalVars - 1
            Jma[inx_D+14+3*inx,inx_D+14+3*inx] += dTa_D / (HUB*a_val)
            Jma[inx_D+14+3*inx,inx_D+14+3*inx-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[inx_D+14+3*inx,inx_D+14+3*inx+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            Jma[inx_D+14+3*inx+1,inx_D+14+3*inx+1] += dTa_D / (HUB*a_val)
            Jma[inx_D+14+3*inx+1,inx_D+14+3*inx+1-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[inx_D+14+3*inx+1,inx_D+14+3*inx+1+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))

            Jma[inx_D+14+3*inx+2,inx_D+14+3*inx+2-3] += self.k*elV/((2.*elV + 1.)*(HUB*a_val))
            Jma[inx_D+14+3*inx+2,inx_D+14+3*inx+2+3] += -self.k*(elV+1.)/((2.*elV + 1.)*(HUB*a_val))


        # Theta Lmax
        Jma[self.TotalVars-3, self.TotalVars-3-3] += self.k / (HUB*a_val)
        Jma[self.TotalVars-3, self.TotalVars-3] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val)

        Jma[-3, -3-3] += self.k / (HUB*a_val)
        Jma[-3, -3] += (-(self.Lmax+1.)/eta + dTa_D) / (HUB*a_val)

        # Theta Lmax
        Jma[self.TotalVars-2, self.TotalVars-2-3] += self.k / (HUB*a_val)
        Jma[self.TotalVars-2, self.TotalVars-2] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val)

        Jma[-2, -2-3] += self.k / (HUB*a_val)
        Jma[-2, -2] += (-(self.Lmax+1.)/eta + dTa_D) / (HUB*a_val)

        # Theta Lmax
        Jma[self.TotalVars-1, self.TotalVars-1-3] += self.k / (HUB*a_val)
        Jma[self.TotalVars-1, self.TotalVars-1] += -(self.Lmax+1.)/(eta*HUB*a_val)

        Jma[-1, -1-3] += self.k / (HUB*a_val)
        Jma[-1, -1] += -(self.Lmax+1.)/(eta*HUB*a_val)

        return Jma

#
#    def Csnd(self, a):
#        return self.Csnd_interp(np.log10(a))/a

    def scale_a(self, eta):
        return 10.**self.ct_to_scale(np.log10(eta))
    
    def conform_T(self, a):
        return quad(lambda x: 1./self.H_0 /np.sqrt(self.omega_R_T+self.omega_M_T*x+self.omega_L_T*x**4.), 0., a)[0]
    
    def hubble(self, a):
        return self.H_0*np.sqrt(self.omega_R_T*a**-4+self.omega_M_T*a**-3.+self.omega_L_T)

    def xe_deta(self, a):
        return 10.**self.Xe(np.log10(a))

    def rhoCDM(self, a):
        return self.omega_cdm_T * self.H_0**2. * a**-3.
    
    def rhoCDM_Indiv(self, a, uni=0):
        return self.omega_cdm[uni] * self.H_0**2. * a**-3.

    def rhoB(self, a):
        return self.omega_b_T * self.H_0**2. * a**-3.

    def rhoB_Indiv(self, a, uni=0):
        return self.omega_b[uni] * self.H_0**2. * a**-3.

    def rhoG(self, a):
        return self.omega_g_T * self.H_0**2. * a**-4.

    def rhoG_Indiv(self, a, uni=0):
        return self.omega_g[uni] * self.H_0**2. * a**-4.

    def rhoNeu(self, a):
        return self.omega_nu_T * self.H_0**2. * a**-4.

    def rhoNeu_Indiv(self, a, uni=0):
        return self.omega_nu[uni] * self.H_0**2. * a**-4.

    def epsilon_test(self, a):
        denom = (self.omega_M_T*a**-3. + self.omega_R_T*a**-4. + self.omega_L_T)
        
        phiTerm = -2./3.*(self.k/(a*self.H_0))**2.*self.combined_vector[0][-1]
        denTerm = (self.omega_cdm[0]*self.combined_vector[1][-1]+self.omega_b[0]*self.combined_vector[3][-1])*a**-3. +\
                  4.*(self.omega_g[0]*self.combined_vector[5][-1]+self.omega_nu[0]*self.combined_vector[7][-1])*a**-4.
        denTerm_D = (self.omega_cdm[1]*self.combined_vector[self.TotalVars][-1]+
                     self.omega_b[1]*self.combined_vector[self.TotalVars+2][-1])*a**-3. +\
                  4.*(self.omega_g[1]*self.combined_vector[self.TotalVars+4][-1]+
                    self.omega_nu[1]*self.combined_vector[self.TotalVars+6][-1])*a**-4.
        
        velTerm = 3.*a*self.hubble(a)/self.k*(
                 (self.omega_cdm[0]*self.combined_vector[self.TotalVars+1][-1]+
                 self.omega_b[0]*self.combined_vector[self.TotalVars+3][-1])*a**-3. +
                 4.*(self.omega_g[0]*self.combined_vector[self.TotalVars+7][-1]+
                 self.omega_nu[0]*self.combined_vector[self.TotalVars+9][-1])*a**-4.)
        velTerm_D = 3.*a*self.hubble(a)/self.k*(
                 (self.omega_cdm[1]*self.combined_vector[self.TotalVars+1][-1]+
                 self.omega_b[1]*self.combined_vector[self.TotalVars+3][-1])*a**-3. +
                 4.*(self.omega_g[1]*self.combined_vector[self.TotalVars+7][-1]+
                 self.omega_nu[1]*self.combined_vector[self.TotalVars+9][-1])*a**-4.)
        return (phiTerm + denTerm + denTerm_D*self.Nbrane + velTerm + velTerm_D*self.Nbrane)/(denom)

    def save_system(self):
        psi_term = np.zeros(len(self.eta_vector))
        for i in range(len(self.eta_vector)):
            aval = 10.**self.ct_to_scale(np.log10(self.eta_vector[i]))
            psi_term[i] = -12.*(aval**2./self.k**2.)* \
                        ((self.rhoNeu_Indiv(aval, uni=0)*self.combined_vector[13][i] + self.rhoG_Indiv(aval,uni=0)*self.combined_vector[11][i]) + \
                        (self.rhoNeu_Indiv(aval, uni=1)*self.combined_vector[self.TotalVars+12][i] +
                        self.rhoG_Indiv(aval, uni=1)*self.combined_vector[self.TotalVars+10][i])) - \
                         self.combined_vector[0][i]
        
        sve_tab = np.zeros((len(self.eta_vector), 2*self.TotalVars+1))
        sve_tab[:,0] = self.eta_vector
        sve_tab[:,-1] = psi_term
        for i in range(2*self.TotalVars-1):
            sve_tab[:,i+1] = self.combined_vector[i]
        np.savetxt(path + '/OutputFiles/MultiBrane_FieldEvolution_' +
                  '{:.4e}_Nbrane_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}.dat'.format(self.k, self.Nbrane, self.PressureFac, self.ECDM),
                  sve_tab, fmt='%.8e', delimiter='    ')
        
        if self.testing:
            np.savetxt(path+'/OutputFiles/MultiBrane_Background_Nbranes_{:.0e}_PressFac_{:.2e}_eCDM_{:.2e}.dat'.format(self.Nbrane, self.PressureFac, self.ECDM),
                        np.column_stack((self.aLIST, self.etaLIST, self.xeLIST, self.hubLIST, self.csLIST,
                                         self.dtauLIST, self.xeD_LIST, self.csD_LIST, self.dtauD_LIST)))
        return

