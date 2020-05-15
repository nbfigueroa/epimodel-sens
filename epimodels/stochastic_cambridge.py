# -*- coding: utf-8 -*-
from epimodels.utils import *
from epimodels.sims import *
# from sims import *
# from utils import*
import scipy.stats as stats
from scipy.integrate import odeint
from pyross.deterministic import SIR
from typing import Union

import numpy as np
from tqdm import tqdm

class StochasticCambridge():

    def __init__(self, N, M=16, *prob_params, **init_cond):
        #Initialize mandatory parameters

        self.N = N # Total population
        self.M = M # Number of compartments

        # Initialize default values
        self.Ni = N/M*np.ones((M)) # An array representing the breakdown of population by compartments
        #Load from the population demographics file for India
        my_data = np.genfromtxt('data/age_structures/India-2019.csv', delimiter=',', skip_header=1)
        aM, aF = my_data[:, 1], my_data[:, 2]
        self.Ni = aM + aF
        self.Ni = self.Ni[0:self.M]
        self.Ni = self.N*self.Ni/np.sum(self.Ni)
        #self.Ni = 
        
        if self.M != self.Ni.shape[0]:
            self.M = self.Ni.shape[0]
            M = self.M
            Warning('Dimension mismatch. Number of components set as per demographic components')
        
        print(self.M)
        #The default value is equal distribution across all the compartments
        self.Is0 = 1e-6/M * np.ones((M))
        self.Ia0 = np.zeros(M) # Number of asymptomatic patients
        self.R0 = np.zeros(M) #Recovered population
        self.S0 = self.Ni - self.R0 - self.Is0 - self.Ia0
        self.tie_recovery = True #Same recovery rates for symptomatic and asymptomatics patients
        self.alpha = 0 #All patients are symptomatic
        self.fsa = 1 #None of the symptomatics are self-isolating

        #If using the default contact matrices, provide the number of components using the dimensions of the CM
        my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='India',index_col=None)
        self.CH = np.array(my_data)
        my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='India',index_col=None)
        self.CW = np.array(my_data)
        my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='India',index_col=None)
        self.CS = np.array(my_data)
        my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='India',index_col=None)
        self.CO = np.array(my_data)
        

        if not prob_params:
            sim_init_cond = loadSimulationParams(5, scenario=0, plot_data = 0)
            prob_params, _ = getSIRTestingParams(3, 'gamma', **sim_init_cond)
        
        self.multiplier = self.compute_r0({'beta':1, 'gs_inv':7, 'ga_inv':7})
        self.set_distributions(prob_params) #Can be overridden by directly providing distributions
        #Override default arguments
        for key in init_cond:
            setattr(self, key, init_cond[key])
        
        #Check for dimension mismatch between self.M and the loaded CM
        if self.CH.shape[0] != self.M:
            self.M = self.CH.shape[0]
            M = self.M
            Warning('Dimension mismatch. Number of components set as per contact matrics')

    def set_distributions(self, prob_params):
        
        #Currently only supports the 'gamma' distribution, support for other 
        #distributions to be added later. For other distribution directly 
        #provide the scipy.stats random variable object
        
        beta0_loc        = prob_params[1]
        beta0_shape      = prob_params[2]
        beta0_scale      = prob_params[3]
        gamma_inv_loc   = prob_params[4]
        gamma_inv_shape = prob_params[5]
        gamma_inv_scale = prob_params[6]
        
        e_gamma_inv = gamma_inv_loc + gamma_inv_shape*gamma_inv_scale
        multiplier = self.compute_r0({'beta':1, 'gs_inv':7, 'ga_inv':7})
        r0_loc = e_gamma_inv * beta0_loc
        r0_scale = beta0_scale
        r0_shape = e_gamma_inv * beta0_shape
        
        beta_loc = r0_loc/multiplier
        beta_shape = r0_shape/multiplier
        beta_scale = beta0_scale

        #TODO: Outputs only placeholders
        if beta_scale == 0:
            self.beta_dist = stats.norm(loc = beta_loc, scale = 0)
        else:
            self.beta_dist = stats.gamma(a = beta_shape, loc = beta_loc, scale = beta_scale)
        
        if gamma_inv_scale == 0:
            self.gs_inv_dist = stats.norm(loc = gamma_inv_loc, scale = 0)
        else:
            self.gs_inv_dist = stats.gamma(a = gamma_inv_shape, loc = gamma_inv_loc, scale = gamma_inv_scale)
            

        if self.tie_recovery:
            self.ga_inv_dist = self.gs_inv_dist
        else:
            self.ga_inv_dist = stats.gamma(a=1, loc = 0, scale = 1)

    def sample_params(self):

        samples = {}

        if isinstance(self.beta_dist, stats._distn_infrastructure.rv_frozen):
            samples['beta'] = self.beta_dist.rvs(1)[0]
        else:
            samples['beta'] = self.beta_dist

        if isinstance(self.gs_inv_dist, stats._distn_infrastructure.rv_frozen):
            samples['gs_inv'] = self.gs_inv_dist.rvs(1)[0]
        else:
            samples['gs_inv'] = self.gs_inv_dist

        if self.tie_recovery:
            samples['ga_inv'] = samples['gs_inv']
        elif isinstance(self.ga_inv_dist, stats._distn_infrastructure.rv_frozen):
            samples['ga_inv'] = self.ga_inv_dist.rvs(1)[0]
        else:
            samples['ga_inv'] = self.ga_inv_dist

        return samples
    
    def ContactMatrixSteady(self):
        
        def CM(t):
            return self.CH + self.CO + self.CW + self.CS        
        return CM
        
        
    def rollout(self, days, dt = 1, CM_function = None):
        
        if CM_function is None:
            CM_function = self.ContactMatrixSteady()
        
        #Initialize the Cambrdige SIR model with the sampled parameters and simulate
        samples = self.sample_params()
        
        model_params = {}
        model_params['alpha'] = self.alpha
        model_params['beta'] = samples['beta']
        model_params['gIs'] = 1/samples['gs_inv']
        model_params['gIa'] = 1/samples['ga_inv']
        model_params['fsa'] = self.fsa
        #print(model_params)
        #Initialize the mode
        model = SIR(model_params, self.M, self.Ni)
        
        #
        
        #Determine simulation parameters
        #parameters = {'alpha':alpha,'beta':beta, 'gIa':gIa,'gIs':gIs,'fsa':fsa}
        model = SIR(model_params, self.M, self.Ni)
        
        
        #Project the model forward and save the results
        tf = days-1
        nf = days
        output = model.simulate(self.S0, self.Ia0, self.Is0, CM_function, tf, nf)
        
        t = output['t']
        X = output['X']
        M = self.M
        S = np.sum(X[:,0:M], axis=1)
        I = np.sum(X[:, M:3*M])
        R = np.sum(X[:,3*M::])
        
        return (S,I,R), output, samples
        #S = np.sum()
        
    def project(self, days, samples, dt = 1, progbar = True, CM_function = None):
        
        if CM_function is None:
            CM_function = self.ContactMatrixSteady()
        
        S_samples = np.zeros((samples, days))
        I_samples = np.zeros((samples, days))
        R_samples = np.zeros((samples, days))
        if self.tie_recovery:
            param_samples = np.zeros((samples, 2))
        else:
            param_samples = np.zeros((samples, 3))        
        outputs = []
        
        for i in (tqdm(range(samples)) if progbar else range(samples)):
            (S,I,R), output, samples = self.rollout(days, CM_function = CM_function)
            
            #Append all the realization
            S_samples[i,:] = S
            I_samples[i,:] = I
            R_samples[i,:] = R
            if self.tie_recovery:
                param_samples[i,:] = np.array([samples['beta'], samples['gs_inv']])
            else:
                param_samples[i,:] = np.array([samples['beta'], samples['gs_inv'], samples['ga_inv']])
            outputs.append(output)
        
        return (S_samples, I_samples, R_samples), param_samples, outputs
    
    def compute_r0(self, samples: Union[dict, np.array]):
        
        #Handle both dict and array samples
        
        if type(samples) == dict:
            beta = samples['beta']
            gIs = 1/samples['gs_inv']
            gIa = 1/samples['ga_inv']
        else:
            if self.tie_recovery:
                beta = samples[0]
                gIs = 1/samples[1]
                gIa = gIs
            else:
                beta = samples[0]
                gIs = 1/samples[1]
                gIa = 1/samples[2]
        
        C = self.ContactMatrixSteady()(0)
        alpha = self.alpha
        fsa = self.fsa
        M = self.M
        Ni = self.Ni
        
        L0 = np.zeros((M, M))
        L  = np.zeros((2*M, 2*M))
        
        print(self.M)
        print(C.shape)
        print(list(range(M)))
        for i in range(M):
            for j in range(M):
                
                L0[i,j]=C[i,j]*Ni[i]/Ni[j]
        L[0:M, 0:M]     =    alpha*beta/gIs*L0
        L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
        L[M:2*M, 0:M]   =    ((1-alpha)*beta/gIs)*L0
        L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0
        
        return np.real(np.max(np.linalg.eigvals(L)))

        
                                              
    
    
    if __name__ == '__main__':
        
        model = StochasticCambrdige(1, 16)
        data = model.rollout(10)
