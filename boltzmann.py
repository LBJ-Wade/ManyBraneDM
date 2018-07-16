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
        #self.omega_L = omega_L
        self.omega_g = omega_g
        self.omega_nu = omega_nu
        self.omega_M = omega_cdm + omega_b
        self.omega_R = omega_g + omega_nu
        self.omega_L = 1. - self.omega_M - self.omega_R
        self.H_0 = 2.2348e-4 # units Mpc^-1
        self.eta_0 = 1.4387e4

        
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
        
        return

    def compute_funcs(self):
        a0_init = np.logspace(-14, 0, 1e4)
        eta_list = np.zeros_like(a0_init)
        for i in range(len(a0_init)):
            eta_list[i] = self.conform_T(a0_init[i])
        
        self.ct_to_scale = interp1d(np.log10(eta_list), np.log10(a0_init), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(a0_init), np.log10(eta_list), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
            
        self.xe_Tab()
        self.Tab_Temp()

    def clearfiles(self):
        if os.path.isfile(path + '/precomputed/xe_working.dat'):
            os.remove(path + '/precomputed/xe_working.dat')
        if os.path.isfile(path + '/precomputed/tb_working.dat'):
            os.remove(path + '/precomputed/tb_working.dat')

        if os.path.isfile(path + '/precomputed/working_expOpticalDepth.dat'):
            os.remove(path + '/precomputed/working_expOpticalDepth.dat')
        if os.path.isfile(path + '/precomputed/working_VisibilityFunc.dat'):
            os.remove(path + '/precomputed/working_VisibilityFunc.dat')

#    def load_funcs(self):
#        time_table = np.loadtxt(path+'/precomputed/Times_Tables.dat')
#        self.ct_to_scale = interp1d(np.log10(time_table[:,2]), np.log10(time_table[:,1]), kind='linear',
#                                    bounds_error=False, fill_value='extrapolate')
#        self.scale_to_ct = interp1d(np.log10(time_table[:,1]), np.log10(time_table[:,2]), kind='linear',
#                                    bounds_error=False, fill_value='extrapolate')

#        self.dtau_load = np.loadtxt(path + '/precomputed/dtau_CLASS.dat')
#        self.dtau_interp = interp1d(np.log10(self.dtau_load[:,0]), np.log10(self.dtau_load[:,1]), kind='linear',
#                                    bounds_error=False, fill_value='extrapolate')
        #xe_load = np.loadtxt(path + '/precomputed/Xe_evol.dat')
#        xe_load = np.loadtxt(path + '/precomputed/CLASS_xe.dat')
#
#        self.Xe = interp1d(np.log10(xe_load[:,0]), np.log10(xe_load[:,1]), kind='linear', bounds_error=False, fill_value='extrapolate')

#        refst = np.loadtxt('/Users/samuelwitte/Desktop/PBH21/External_tables/CosmoRec.HI.nS_eff_500.nH_3.nHe_3.H_abs.fcorr.Q_lines.X_Recfast.dat')
#        self.Tb = interp1d(np.log10(1./(1.+refst[:,0])), np.log10(refst[:,2]), bounds_error=False, fill_value='extrapolate')

#        cs_load = np.loadtxt(path + '/precomputed/Csound_CLASS.dat')
#        self.Csnd_interp = interp1d(np.log10(cs_load[:,0]), cs_load[:,0]*cs_load[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
#        hubble_load = np.log10(np.loadtxt(path + '/precomputed/Hubble_CT.dat'))
#        self.hubble_CT = interp1d(hubble_load[:,0], hubble_load[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
#        return

    def Tab_Temp(self):
        self.tb_fileNme = path + '/precomputed/tb_working.dat'
        if not os.path.isfile(self.tb_fileNme):
        
            print 'Calculating Tb...'
            y0 = np.log(self.scale_a(np.min([1e-3/self.k, 1e-1/0.7])))
            
            Tg_0 = 2.7255 / np.exp(y0)
            yvals = np.linspace(y0, 0., 1000)
            solvR = odeint(self.dotT, Tg_0, yvals)
            self.Tb_drk = np.column_stack((np.exp(yvals), solvR))
            self.Tb_drk[self.Tb_drk<0] = 1e-20
            np.savetxt(self.tb_fileNme, self.Tb_drk)
        else:
            self.Tb_drk = np.loadtxt(self.tb_fileNme)
        self.Tb = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), bounds_error=False, fill_value='extrapolate')
        return
    
    def dotT(self, T, y):
        kb = 8.617e-5/1e9 # Gev/K
        thompson_xsec = 6.65e-25 # cm^2
        aval = np.exp(y)
        Yp = 0.245
        Mpc_to_cm = 3.086e24
        facxe = 10.**self.Xe(np.log10(aval))
        mol_wei = np.zeros_like(facxe)
        mol_wei[facxe >= 1] = 0.6
        mol_wei[facxe < 1] = 1.22
        n_b = 2.503e-7 / aval**3.
        hub = self.hubble(aval)
        omega_Rat = self.omega_g / self.omega_b
        
        return (-2.*T[0] + (8./3.)*(mol_wei/5.11e-4)*omega_Rat*(facxe*n_b*thompson_xsec/hub)*(2.7255/aval - T[0])*Mpc_to_cm)
    
    def Cs_Sqr(self, a):
        kb = 8.617e-5/1e9 # GeV/K
        facxe = 10.**self.Xe(np.log10(a))
        mol_wei = np.zeros_like(facxe)
        mol_wei[facxe >= 1] = 0.6
        mol_wei[facxe < 1] = 1.22
        Tb = 10.**self.Tb(np.log10(a))
        extraPT = self.dotT([Tb], np.log(a)) * a
        
        return kb*Tb/mol_wei*(1. - 1./3. * extraPT/Tb)
    
    def xe_Tab(self):
        self.Xe_fileNme = path + '/precomputed/xe_working.dat'
        if not os.path.isfile(self.Xe_fileNme):
            tcmbD=2.7255
            print 'Calculating Free Electron Fraction'
            x0 = 1.
            yvals = np.linspace(3.5, 0, 500)
            solvR = odeint(self.xeDiff, x0, yvals)
            x0Convert = 1. / (1. + 10.**yvals)
 
            self.Xe_dark = np.column_stack((x0Convert, solvR[:,0]*1.079))
            np.savetxt(self.Xe_fileNme, self.Xe_dark)
        else:
            self.Xe_dark = np.loadtxt(self.Xe_fileNme)
        self.Xe = interp1d(np.log10(self.Xe_dark[:,0]), np.log10(self.Xe_dark[:,1]), bounds_error=False, fill_value='extrapolate')
        return
    
    def xeDiff(self, val, y, hydrogen=True, first=True):
        yy = 10.**y
        tcmbD=2.7255
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
        Yp = 0.245
        
        n_b = 2.503e-7 / aval**3.
        Tg = tcmbD * (1. + yy)
        hub = self.hubble(aval)
        FScsnt = 7.29e-3
        
        alpha2 = 9.78*(FScsnt/me)**2.*np.sqrt(ep0/(kb*tcmbD*(1.+yy)))*np.log(ep0/(kb*tcmbD*(1.+yy)))/(GeV_cm**2.) # cm^2
        beta = alpha2*(me*(kb*tcmbD*(1.+yy))/(2.*np.pi))**(3./2.)*np.exp(-ep0/(kb*tcmbD*(1.+yy)))*GeV_cm**3. # 1/cm
        beta2 = alpha2*(me*(kb*tcmbD*(1.+yy))/(2.*np.pi))**(3./2.)*np.exp(-ep0/(4.*kb*tcmbD*(1.+yy)))*GeV_cm**3.*Mpc_to_cm
        
        if val[0] > 0.999:
            Cr = 1.
        else:
            Lalpha = (3.*ep0)**3.*hub / (64.*np.pi**2*(1-val[0])*n_b*Yp) * GeV_cm**3.
            L2g = 8.227 / 2.998e10 * Mpc_to_cm
            Cr = (Lalpha + L2g) / (Lalpha + L2g + beta2)
        
        Value = Cr*np.log(10.)*yy*(-aval)/(hub)*((1-val[0])*beta - val[0]**2.*n_b*alpha2)*Mpc_to_cm
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
            
            np.savetxt(self.fileN_optdep, np.column_stack((avals, np.exp(-tau))))
            np.savetxt(self.fileN_visibil, np.column_stack((avals, -dtau * np.exp(-tau))))
    
        return

    def init_conds(self, eta_0, aval):
        OM = self.omega_M * self.H_0**2./self.hubble(aval)**2./aval**3.
        OR = self.omega_R * self.H_0**2./self.hubble(aval)**2./aval**4.
        ONu = self.omega_nu * self.H_0**2./self.hubble(aval)**2./aval**4.
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
        
        # TESTING
#        aL = np.logspace(-9, 0, 300)
#        xeV = 10.**self.Xe(np.log10(aL))
#        csT = self.Cs_Sqr(aL)
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
                #print 'Failed test of expansion/Courant time...'
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
        
        tflip_TCA = 1e-4
        
        PsiTerm = np.zeros(self.TotalVars)
        PsiTerm[0] = -1.
        PsiTerm[11] = -12.*(a_val/self.k)**2.*self.rhoG(a_val)
        PsiTerm[13] = -12.*(a_val/self.k)**2.*self.rhoNeu(a_val)
        
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
        Jma[11,6] += - dTa / (10.*HUB*a_val)
        Jma[11,12] += - dTa /(10.*HUB*a_val)
        # ThetaP 2
        Jma[12,9] += 2.*self.k / (5.*HUB*a_val)
        Jma[12,15] += -3.*self.k / (5.*HUB*a_val)
        Jma[12,12] += 9.*dTa / (10.*HUB*a_val)
        Jma[12,11] += -dTa / (10.*HUB*a_val)
        Jma[12,6] += - dTa / (10.*HUB*a_val)
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
        # Theta Lmax
        Jma[-2, -2-3] += self.k / (HUB*a_val)
        Jma[-2, -2] += (-(self.Lmax+1.)/eta + dTa) / (HUB*a_val)
        # Theta Lmax
        Jma[-1, -1-3] += self.k / (HUB*a_val)
        Jma[-1, -1] += -(self.Lmax+1.)/(eta*HUB*a_val)
        return Jma

#    def Csnd(self, a):
#        return self.Csnd_interp(np.log10(a))/a

    def scale_a(self, eta):
        return 10.**self.ct_to_scale(np.log10(eta))
    
    def conform_T(self, a):
        return quad(lambda x: 1./self.H_0 /np.sqrt(self.omega_R+self.omega_M*x+self.omega_L*x**4.), 0., a)[0]
        #return 10.**self.scale_to_ct(np.log10(a))

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
        
        #self.yp_prime = 0.2262 + 0.0135*np.log(self.omega_b[1]/self.omega_b[0]*6.25)
        ngamma_pr = 410.7 * (self.darkCMB_T/2.7255)**3.
        nbarys = 2.503e-7 * (omega_b[1]/omega_b[0])
        etaPr = 6.1e-10 * (omega_b[1]/omega_b[0])*(omega_g[0]/omega_g[1])
        self.yp_prime = Yp_Prime(etaPr)
        #print self.omega_M_T+ self.omega_cdm_T + self.omega_R_T + self.omega_L_T
        
        print 'Fraction of baryons on each brane: {:.3f}'.format(omega_b[1]/omega_b[0])
        
        self.H_0 = 2.2348e-4 # units Mpc^-1
        self.eta_0 = 1.4387e4

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
        
        self.ct_to_scale = interp1d(np.log10(eta_list), np.log10(a0_init), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(a0_init), np.log10(eta_list), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')

        self.xe_Tab()
        self.Tab_Temp()
        self.xeDark_Tab()
        self.Tab_dark_Temp()
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
                                    
#    def load_funcs(self):
#        time_table = np.loadtxt(path+'/precomputed/Times_Tables.dat')
#        self.ct_to_scale = interp1d(np.log10(time_table[:,2]), np.log10(time_table[:,1]), kind='linear',
#                                    bounds_error=False, fill_value='extrapolate')
#        self.scale_to_ct = interp1d(np.log10(time_table[:,1]), np.log10(time_table[:,2]), kind='linear',
#                                    bounds_error=False, fill_value='extrapolate')

#        self.dtau_load = np.loadtxt(path + '/precomputed/dtau_CLASS.dat')
#        self.dtau_interp = interp1d(np.log10(self.dtau_load[:,0]), np.log10(self.dtau_load[:,1]), kind='linear',
#                                    bounds_error=False, fill_value='extrapolate')
        #xe_load = np.loadtxt(path + '/precomputed/Xe_evol.dat')
#        xe_load = np.loadtxt(path + '/precomputed/CLASS_xe.dat')
#        self.Xe = interp1d(np.log10(xe_load[:,0]), xe_load[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
#        cs_load = np.loadtxt(path + '/precomputed/Csound_CLASS.dat')
#        self.Csnd_interp = interp1d(np.log10(cs_load[:,0]), cs_load[:,0]*cs_load[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
#        hubble_load = np.log10(np.loadtxt(path + '/precomputed/Hubble_CT.dat'))
#        self.hubble_CT = interp1d(hubble_load[:,0], hubble_load[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
#        return

    def Tab_Temp(self):
        self.tb_fileNme = path + '/precomputed/tb_working.dat'
        if not os.path.isfile(self.tb_fileNme):
        
            print 'Calculating Tb...'
            y0 = np.log(self.scale_a(np.min([1e-3/self.k, 1e-1/0.7])))
            
            Tg_0 = 2.7255 / np.exp(y0)
            yvals = np.linspace(y0, 0., 1000)
            solvR = odeint(self.dotT, Tg_0, yvals)
            self.Tb_drk = np.column_stack((np.exp(yvals), solvR))
            self.Tb_drk[self.Tb_drk<0] = 1e-20
            np.savetxt(self.tb_fileNme, self.Tb_drk)
        else:
            self.Tb_drk = np.loadtxt(self.tb_fileNme)
        self.Tb = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), bounds_error=False, fill_value='extrapolate')
        return
    
    def dotT(self, T, y):
        kb = 8.617e-5/1e9 # Gev/K
        thompson_xsec = 6.65e-25 # cm^2
        aval = np.exp(y)
        Yp = 0.245
        Mpc_to_cm = 3.086e24
        facxe = 10.**self.Xe(np.log10(aval))
        mol_wei = np.zeros_like(facxe)
        mol_wei[facxe >= 1] = 0.6
        mol_wei[facxe < 1] = 1.22
        n_b = 2.503e-7 / aval**3.
        hub = self.hubble(aval)
        omega_Rat = self.omega_g[0] / self.omega_b[0]
        
        return (-2.*T[0] + (8./3.)*(mol_wei/5.11e-4)*omega_Rat*(facxe*n_b*thompson_xsec/hub)*(2.7255/aval - T[0])*Mpc_to_cm)
    
    def Cs_Sqr(self, a):
        kb = 8.617e-5/1e9 # GeV/K
        facxe = 10.**self.Xe(np.log10(a))
        mol_wei = np.zeros_like(facxe)
        mol_wei[facxe >= 1] = 0.6
        mol_wei[facxe < 1] = 1.22
        Tb = 10.**self.Tb(np.log10(a))
        extraPT = self.dotT([Tb], np.log(a)) * a
        
        return kb*Tb/mol_wei*(1. - 1./3. * extraPT/Tb)
    
    def xe_Tab(self):
        self.Xe_fileNme = path + '/precomputed/xe_working.dat'
        if not os.path.isfile(self.Xe_fileNme):
            tcmbD=2.7255
            print 'Calculating Free Electron Fraction'
            x0 = 1.
            yvals = np.linspace(3.5, -1, 500)
            solvR = odeint(self.xeDiff, x0, yvals)
            x0Convert = 1. / (1. + 10.**yvals)
 
            self.Xe_dark = np.column_stack((x0Convert, solvR[:,0]*1.079))
            np.savetxt(self.Xe_fileNme, self.Xe_dark)
        else:
            self.Xe_dark = np.loadtxt(self.Xe_fileNme)
        self.Xe = interp1d(np.log10(self.Xe_dark[:,0]), np.log10(self.Xe_dark[:,1]), bounds_error=False, fill_value='extrapolate')
        return
    
    def xeDiff(self, val, y, hydrogen=True, first=True):
        yy = 10.**y
        tcmbD=2.7255
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
        Yp = 0.245
        
        n_b = 2.503e-7 / aval**3.
        Tg = tcmbD * (1. + yy)
        hub = self.hubble(aval)
        FScsnt = 7.29e-3
        
        alpha2 = 9.78*(FScsnt/me)**2.*np.sqrt(ep0/(kb*tcmbD*(1.+yy)))*np.log(ep0/(kb*tcmbD*(1.+yy)))/(GeV_cm**2.) # cm^2
        beta = alpha2*(me*(kb*tcmbD*(1.+yy))/(2.*np.pi))**(3./2.)*np.exp(-ep0/(kb*tcmbD*(1.+yy)))*GeV_cm**3. # 1/cm
        beta2 = alpha2*(me*(kb*tcmbD*(1.+yy))/(2.*np.pi))**(3./2.)*np.exp(-ep0/(4.*kb*tcmbD*(1.+yy)))*GeV_cm**3.*Mpc_to_cm
        
        if val[0] > 0.999:
            Cr = 1.
        else:
            Lalpha = (3.*ep0)**3.*hub / (64.*np.pi**2*(1-val[0])*n_b*Yp) * GeV_cm**3.
            L2g = 8.227 / 2.998e10 * Mpc_to_cm
            Cr = (Lalpha + L2g) / (Lalpha + L2g + beta2)
        
        Value = Cr*np.log(10.)*yy*(-aval)/(hub)*((1-val[0])*beta - val[0]**2.*n_b*alpha2)*Mpc_to_cm
        return [Value]
        
    
    def Tab_dark_Temp(self):
        self.tbDk_fileNme = path + '/precomputed/tb_dark_working.dat'
        if not os.path.isfile(self.tbDk_fileNme):
            print 'Calculating Tb evolution of Dark Brane...'
            y0 = np.log(self.scale_a(np.min([1e-3/self.k, 1e-1/0.7])))

            Tg_0 = self.darkCMB_T / np.exp(y0)
            yvals = np.linspace(y0, 0., 1000)
            solvR = odeint(self.dotT_Dk, Tg_0, yvals)
            self.Tb_drk = np.column_stack((np.exp(yvals), solvR))
            #np.savetxt(path + '/precomputed/TEST_DARK_TEMP.dat',self.Tb_drk)
            self.Tb_drk[self.Tb_drk<0] = 1e-20
            np.savetxt(self.tbDk_fileNme, self.Tb_drk)
        else:
            self.Tb_drk = np.loadtxt(self.tbDk_fileNme)
        
        self.Tb_DARK = interp1d(np.log10(self.Tb_drk[:,0]), np.log10(self.Tb_drk[:,1]), bounds_error=False, fill_value='extrapolate')
        return
    
    def dotT_Dk(self, T, y):
        kb = 8.617e-5/1e9 # Gev/K
        aval = np.exp(y)
        Yp = self.yp_prime
        Mpc_to_cm = 3.086e24
        facxe = 10.**self.XE_DARK_B(np.log10(aval))
        mol_wei = (0.7 * facxe + (1. - facxe)*1.735) * 0.938
        n_b = 2.503e-7 / aval**3. * self.omega_b[1]/self.omega_b[0]
        hub = self.hubble(aval)
        omega_Rat = self.omega_g[1]/self.omega_b[1]
        xeval = 10.**self.XE_DARK_B(np.log10(aval))
        return (-2.*T[0] + (8./3.)*(mol_wei/5.11e-4)*omega_Rat*(xeval*n_b*6.65e-25/hub)*(self.darkCMB_T/aval - T[0])*Mpc_to_cm)
    
    def Cs_Sqr_Dark(self, a):
        kb = 1.3806e-23
        facxe = 10.**self.XE_DARK_B(np.log10(a))
        mol_wei = (0.7 * facxe + (1. - facxe)*1.735) * 0.938
        Tb = 10.**self.Tb_DARK(np.log10(a))
        extraPT = self.dotT([Tb], np.log(a)) / a
        csSq = 2.998e8**2.
        return kb*Tb/mol_wei*(1./3. - extraPT/Tb)/csSq
    
    
    
    def xeDark_Tab(self):
        tcmbD=self.darkCMB_T
        self.Xedk_fileNme = path + '/precomputed/xe_dark_working.dat'
        if not os.path.isfile(self.Xedk_fileNme):
            print 'Calculating Dark Free Electron Fraction'
            x0 = 1.
            yvals = np.linspace(3.5, -1, 500)
            solvR = odeint(self.xeDiff, x0, yvals)
            x0Convert = 1. / (1. + 10.**yvals)
 
            self.Xe_dark2 = np.column_stack((x0Convert, solvR[:,0]*1.079))
            np.savetxt(self.Xedk_fileNme, self.Xe_dark2)
        else:
            self.Xe_dark2 = np.loadtxt(self.Xedk_fileNme)
        self.XE_DARK_B = interp1d(np.log10(self.Xe_dark2[:,0]), np.log10(self.Xe_dark2[:,1]), bounds_error=False, fill_value='extrapolate')
        return
    
    def xeDiff_Dk(self, val, y, hydrogen=True, first=True):
        yy = 10.**y
        tcmbD=self.darkCMB_T
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
        Yp=self.yp_prime
        
        n_b = 2.503e-7 / aval**3. * self.omega_b[1]/self.omega_b[0]
        Tg = tcmbD * (1. + yy)
        hub = self.hubble(aval)
        FScsnt = 7.29e-3
        
        alpha2 = 9.78*(FScsnt/me)**2.*np.sqrt(ep0/(kb*tcmbD*(1.+yy)))*np.log(ep0/(kb*tcmbD*(1.+yy)))/(GeV_cm**2.) # cm^2
        beta = alpha2*(me*(kb*tcmbD*(1.+yy))/(2.*np.pi))**(3./2.)*np.exp(-ep0/(kb*tcmbD*(1.+yy)))*GeV_cm**3. # 1/cm
        beta2 = alpha2*(me*(kb*tcmbD*(1.+yy))/(2.*np.pi))**(3./2.)*np.exp(-ep0/(4.*kb*tcmbD*(1.+yy)))*GeV_cm**3.*Mpc_to_cm
        
        if val[0] > 0.999:
            Cr = 1.
        else:
            Lalpha = (3.*ep0)**3.*hub / (64.*np.pi**2*(1-val[0])*n_b*Yp) * GeV_cm**3.
            L2g = 8.227 / 2.998e10 * Mpc_to_cm
            Cr = (Lalpha + L2g) / (Lalpha + L2g + beta2)
        
        Value = Cr*np.log(10.)*yy*(-aval)/(hub)*((1-val[0])*beta - val[0]**2.*n_b*alpha2)*Mpc_to_cm
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
        
        CsndB = self.Cs_Sqr(a_val)
        CsndB_D = self.Cs_Sqr_Dark(a_val)
        
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

