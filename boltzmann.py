import numpy as np
import os
import sympy
from sympy import *
from sympy.matrices import *
import scipy
import scipy.linalg
from scipy.linalg import lu_solve, lu_factor
from scipy.integrate import ode, quad
from scipy.interpolate import interp1d
from constants import *
import time

path = os.getcwd()


class Universe(object):

    def __init__(self, k, omega_b, omega_cdm, omega_g, omega_L, omega_nu, accuracy=4e-3, stepsize=0.05):
        self.omega_b = omega_b
        self.omega_cdm = omega_cdm
        self.omega_L = omega_L
        self.omega_g = omega_g
        self.omega_nu = omega_nu
        self.omega_M = omega_cdm + omega_b
        self.omega_R = omega_g + omega_nu
        self.omega_L = 1. - self.omega_M - self.omega_R
        self.H_0 = 2.2348e-4 # units Mpc^-1
        self.eta_0 = 1.4100e4 #1.4135e+04


        self.stepsize = stepsize
        
        self.k = k
        print 'Solving perturbations for k = {:.3e} \n'.format(k)
        #self.n_index = 0.967
        
        self.accuracy = accuracy
        self.TotalVars = 23
        self.step = 0
        
        self.combined_vector = np.zeros(23 ,dtype=object)
        self.Psi_vec = []
        self.combined_vector[0] = self.Phi_vec = []
        self.combined_vector[1] = self.dot_rhoCDM_vec = []
        self.combined_vector[2] = self.dot_velCDM_vec = []
        self.combined_vector[3] = self.dot_rhoB_vec = []
        self.combined_vector[4] = self.dot_velB_vec = []
        self.combined_vector[5] = self.dot_Theta_0_vec = []
        self.combined_vector[6] = self.dot_ThetaP_0_vec = []
        self.combined_vector[7] = self.dot_N_0_vec = []
        self.combined_vector[8] = self.dot_Theta_1_vec = []
        self.combined_vector[9] = self.dot_ThetaP_1_vec = []
        self.combined_vector[10] = self.dot_N_1_vec = []
        self.combined_vector[11] = self.dot_Theta_2_vec = []
        self.combined_vector[12] = self.dot_ThetaP_2_vec = []
        self.combined_vector[13] = self.dot_N_2_vec = []
        self.combined_vector[14] = self.dot_Theta_3_vec = []
        self.combined_vector[15] = self.dot_ThetaP_3_vec = []
        self.combined_vector[16] = self.dot_N_3_vec = []
        self.combined_vector[17] = self.dot_Theta_4_vec = []
        self.combined_vector[18] = self.dot_ThetaP_4_vec = []
        self.combined_vector[19] = self.dot_N_4_vec = []
        self.combined_vector[20] = self.dot_Theta_5_vec = []
        self.combined_vector[21] = self.dot_ThetaP_5_vec = []
        self.combined_vector[22] = self.dot_N_5_vec = []
        
        self.load_funcs()

    def load_funcs(self):
        time_table = np.loadtxt(path+'/precomputed/Times_Tables.dat')
        self.ct_to_scale = interp1d(np.log10(time_table[:,2]), np.log10(time_table[:,1]), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self.scale_to_ct = interp1d(np.log10(time_table[:,1]), np.log10(time_table[:,2]), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
                                    
        self.dtau_load = np.loadtxt(path + '/precomputed/dtau_CLASS.dat')
        self.dtau_interp = interp1d(np.log10(self.dtau_load[:,0]), np.log10(self.dtau_load[:,1]), kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        xe_load = np.log10(np.loadtxt(path + '/precomputed/Xe_evol.dat'))
        self.Xe = interp1d(xe_load[:,0], xe_load[:,1], kind='linear', bounds_error=False, fill_value='extrapolate') # func \eta
        cs_load = np.loadtxt(path + '/precomputed/Csound_CLASS.dat')
        self.Csnd_interp = interp1d(np.log10(cs_load[:,0]), np.log10(cs_load[:,1]), kind='linear',
                             bounds_error=False, fill_value='extrapolate')
        hubble_load = np.log10(np.loadtxt(path + '/precomputed/Hubble_CT.dat'))
        self.hubble_CT = interp1d(hubble_load[:,0], hubble_load[:,1], kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
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
            self.dot_Theta_0_vec.append(-1./2.*self.Psi_vec[-1])
            self.dot_Theta_1_vec.append(1./6*eta_0*self.k*self.Psi_vec[-1])
            self.dot_Theta_2_vec.append(0.)
            self.dot_Theta_3_vec.append(0.)
            self.dot_Theta_4_vec.append(0.)
            self.dot_Theta_5_vec.append(0.)
            self.dot_ThetaP_0_vec.append(0.)
            self.dot_ThetaP_1_vec.append(0.)
            self.dot_ThetaP_2_vec.append(0.)
            self.dot_ThetaP_3_vec.append(0.)
            self.dot_ThetaP_4_vec.append(0.)
            self.dot_ThetaP_5_vec.append(0.)
            self.dot_N_0_vec.append(-1./2.*self.Psi_vec[-1])
            self.dot_N_1_vec.append(1./6*eta_0*self.k*self.Psi_vec[-1])
            #self.dot_N_2_vec.append(1./30.*self.k*eta_0*self.Psi_vec[-1])
            self.dot_N_2_vec.append((self.k/aval)**2/30.*rfactor/self.rhoNeu(aval)*self.Psi_vec[-1])
            self.dot_N_3_vec.append(0.)
            self.dot_N_4_vec.append(0.)
            self.dot_N_5_vec.append(0.)


#
#        self.init_conds = [-(1.+2.*rfactor/5.)*self.Psi_vec[-1], -3./2.*self.Psi_vec[-1],
#                        1./2*eta_0*self.k*self.Psi_vec[-1], -3./2.*self.Psi_vec[-1],
#                        1./2*eta_0*self.k*self.Psi_vec[-1], -1./2.*self.Psi_vec[-1],
#                        1./6*eta_0*self.k*self.Psi_vec[-1], 0., 0., 0., 0., 0., 0.,
#                        0., 0., 0., 0., -1./2.*self.Psi_vec[-1], 1./6*eta_0*self.k*self.Psi_vec[-1],
#                        1./30.*self.k*eta_0*self.Psi_vec[-1], 0., 0., 0.]
        self.step = 0
        return
    
    
    def solve_system(self):
        eta_st = np.min([1e-3/self.k, 1e-1/0.7]) # Initial conformal time in Mpc
        y_st = np.log(self.scale_a(eta_st))
        eta_st = self.conform_T(np.exp(y_st))
        
        self.init_conds(eta_st, np.exp(y_st))
        self.eta_vector = [eta_st]
        self.y_vector = [y_st]
        
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
            
            if self.step%1000 == 0:
                print 'Last a: {:.7e}, New a: {:.7e}'.format(np.exp(self.y_vector[-2]), np.exp(self.y_vector[-1]))
            if ((y_diff > eta_use*np.exp(y_use)*self.hubble(np.exp(y_use))) or
                (y_diff > np.exp(y_use)*self.hubble(np.exp(y_use))/self.k)):
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                print 'Failed test of expansion/Courant time...'
                try_count += 1
                continue
            self.step_solver()
            
#            r = ode(self.ode_system).set_integrator('dopri5')
#            r.set_initial_value(self.init_conds, self.y_vector[0])
#            t1 = 0.
#            dt = 1e-5
#            while r.successful() and r.t < t1:
#                r.integrate(r.t+dt)
#                print r.t+dt
#            print time.time() - t0
#            exit()

            test_epsilon = self.epsilon_test(np.exp(self.y_vector[-1]))
            #print test_epsilon, 'EPS'
            if np.abs(test_epsilon) > self.accuracy:
                raise ValueError
                self.stepsize *= 0.5
                self.eta_vector.pop()
                self.y_vector.pop()
                print 'Failed epsilon test...   Value: {:.3e}'.format(test_epsilon)
                try_count += 1
                continue
            self.step += 1
            if (np.abs(test_epsilon) < 1e-7*self.accuracy) and not last_step_up:
                self.stepsize *= 1.1
                last_step_up = True
                print 'Increase Step Size'
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
        Ident = eye(23)
        Jmat = self.matrix_J(self.y_vector[-1])
        Amat = (1.+2.*tau_n)/(1.+tau_n)*Ident - delt*Jmat
        bvec = self.b_vector(tau_n)
        LUpiv = lu_factor(Amat)
        ysol = lu_solve(LUpiv, bvec)
        
        for i in range(self.TotalVars):
            self.combined_vector[i].append(ysol[i][0])
        
        #aval = np.exp(self.y_vector[-1])
#        print -12.*(aval**2./self.k**2.*(self.rhoNeu(aval)*self.combined_vector[13][-1] +
#                                                       self.rhoG(aval)*self.combined_vector[11][-1])) - self.combined_vector[0][-1]

        return
    
    def b_vector(self, tau):
        bvec = zeros(23,1)
        for i in range(self.TotalVars):
            if self.step == 0:
                #bvec[i] = self.combined_vector[i][-1]
                bvec[i] = (1.+tau)*self.combined_vector[i][-1]
            else:
                bvec[i] = (1.+tau)*self.combined_vector[i][-1] - tau**2./(1.+tau)*self.combined_vector[i][-2]
        return bvec
    
#    def ode_system(self, yval, yvec):
#        jmat = self.matrix_J(yval)
#        soln = np.zeros_like(yvec)
#        for i in range(len(yvec)):
#            for j in range(len(yvec)):
#                soln[i] += jmat[i,j]*yvec[j]
#        return soln

    def matrix_J(self, z_val):
        a_val = np.exp(z_val)
        eta = self.conform_T(a_val)
        Jma = zeros(23,23)
        Rfac = (3.*self.rhoB(a_val))/(4.*self.rhoG(a_val))
        RR = (4.*self.rhoG(a_val))/(3.*self.rhoB(a_val))
        HUB = self.hubble(a_val)
        dTa = -self.xe_deta(eta)*(1.-0.245)*2.503e-7*6.65e-29*1e4/a_val**2./3.24078e-25
        CsndB = self.Csnd(eta)
        
        tflip_TCA = 4e-4
        tflip_HO = 1e-8
        
        PsiTerm = zeros(1,23)
        PsiTerm[0,0] = -1.
        PsiTerm[0,11] = -12.*(a_val/self.k)**2.*self.rhoG(a_val)
        PsiTerm[0,13] = -12.*(a_val/self.k)**2.*self.rhoNeu(a_val)
        
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
            Jma[8,11] += -2.*self.k/(3.*HUB*a_val)
            Jma[8,:] += self.k * PsiTerm / (3.*HUB*a_val)
        else:
            Jma[8,4] += -1./(3.*RR)
            Jma[8,3] += CsndB*self.k/(HUB*a_val*RR*3.)
            Jma[8,5] += self.k/(3.*HUB*a_val)
            Jma[8,11] += -self.k/(6.*HUB*a_val)
            Jma[8,:] += (1.+RR)*self.k/(3.*RR*HUB*a_val)*PsiTerm
            Jma[8,:] += -Jma[4,:]/(3.*RR)
        
        # ThetaP 1
        Jma[9,6] = self.k / (3.*HUB*a_val)
        Jma[9,12] = -2.*self.k / (3.*HUB*a_val)
        Jma[9,9] = dTa / (HUB*a_val)
        # Neu 1
        Jma[10,7] += self.k / (3.*HUB*a_val)
        Jma[10,13] += -2.*self.k/ (3.*HUB*a_val)
        Jma[10,:] += self.k * PsiTerm / (3.*HUB*a_val)
        
        # Theta 2
        #if a_val > tflip_HO:
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
        Jma[12,6] += -dTa / (10.*HUB*a_val)
        # Neu 2
        Jma[13,10] += 2.*self.k / (5.*HUB*a_val)
        Jma[13,16] += -3.*self.k / (5.*HUB*a_val)
        # Theta 3
        Jma[14,14] += dTa / (HUB*a_val)
        Jma[14,11] += 3.*self.k / (7.*HUB*a_val)
        Jma[14,17] += -4.*self.k/ (7.*HUB*a_val)
        # ThetaP 3
        Jma[15,15] += dTa / (HUB*a_val)
        Jma[15,12] += 3.*self.k/ (7.*HUB*a_val)
        Jma[15,18] += -4.*self.k/ (7.*HUB*a_val)
        # Neu 3
        Jma[16,13] += 3.*self.k/ (7.*HUB*a_val)
        Jma[16,19] += -4*self.k/ (7.*HUB*a_val)
        # Theta 4
        Jma[17,17] += dTa / (HUB*a_val)
        Jma[17,14] += 4.*self.k / (9.*HUB*a_val)
        Jma[17,20] += -5.*self.k / (9.*HUB*a_val)
        # ThetaP 4
        Jma[18,18] += dTa / (HUB*a_val)
        Jma[18,15] += 4.*self.k / (9.*HUB*a_val)
        Jma[18,21] += -5.*self.k / (9.*HUB*a_val)
        # Neu 4
        Jma[19,16] += 4.*self.k / (9.*HUB*a_val)
        Jma[19,22] += -5.*self.k / (9.*HUB*a_val)
        # Theta 5
        Jma[20, 17] += self.k / (HUB*a_val)
        Jma[20, 20] += (-6./eta + dTa) / (HUB*a_val)
        # ThetaP 5
        Jma[21, 18] += self.k / (HUB*a_val)
        Jma[21, 21] += (-6./eta + dTa) / (HUB*a_val)
        # Neu 5
        Jma[22, 19] += self.k / (HUB*a_val)
        Jma[22, 22] += -6./(eta*HUB*a_val)
        return Jma

    def Csnd(self, eta):
        return 10.**self.Csnd_interp(np.log10(eta))

    def dtau_deta(self, a):
        if a > self.dtau_load[0,0]:
            return self.dtau_load[0,1]
        if a < self.dtau_load[-1,0]:
            return self.dtau_load[-1,1]
        else:
            return 10.**self.dtau_interp(np.log10(a))

    def scale_a(self, eta):
        return 10.**self.ct_to_scale(np.log10(eta))
    
    def conform_T(self, a):
        return quad(lambda x: 1./self.H_0 /np.sqrt(self.omega_R+self.omega_M*x+self.omega_L*x**4.), 0., a)[0]
        #return 10.**self.scale_to_ct(np.log10(a))

    def hubble(self, a):
        return self.H_0*np.sqrt(self.omega_R*a**-4+self.omega_M*a**-3.+self.omega_L)

    def xe_deta(self, eta):
        return 10.**self.Xe(np.log10(eta))

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
        return

