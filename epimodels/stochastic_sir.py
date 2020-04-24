# -*- coding: utf-8 -*-
import scipy.stats as stats
from scipy.integrate  import odeint
import numpy as np
from tqdm import tqdm

from epimodels.utils import *
from epimodels.sims import *


def set_distribution_params(*prob_params):

    # Sample from Gamma Distributions        
    if prob_params[0] == 'gamma':   
        beta_loc        = prob_params[1]
        beta_shape      = prob_params[2]
        beta_scale      = prob_params[3]
        gamma_inv_loc   = prob_params[4]
        gamma_inv_shape = prob_params[5]
        gamma_inv_scale = prob_params[6]
        

        if beta_scale == 0:
            beta_dist      = stats.norm(loc = beta_loc, scale = beta_scale)
        else:        
            beta_dist      = stats.gamma(a = beta_shape, loc = beta_loc, scale = beta_scale)
        
        if gamma_inv_scale == 0:
            gamma_inv_dist = stats.norm(loc = gamma_inv_loc, scale = gamma_inv_scale)
        else:    
            gamma_inv_dist = stats.gamma(a = gamma_inv_shape, loc = gamma_inv_loc, scale = gamma_inv_scale)

    # Sample from LogNormal Distributions        
    if prob_params[0] == 'log-normal':
        beta_mean       = prob_params[1]
        beta_std        = prob_params[2]
        gamma_inv_mean  = prob_params[3]
        gamma_inv_std   = prob_params[4]


        if beta_std == 0:
            beta_dist  = stats.norm(loc = beta_mean, scale = beta_std)
        else:  
            beta_dist  = stats.lognorm(scale = np.exp(beta_mean),s = beta_std)

        # Sample from LogNormal if std not 0
        if gamma_inv_std == 0:
            gamma_inv_dist = stats.norm(loc = gamma_inv_loc, scale = gamma_inv_scale)
        else:  
            gamma_inv_dist = stats.lognorm(scale = np.exp(gamma_inv_mean), s = gamma_inv_std)        

    return beta_dist, gamma_inv_dist


class StochasticSIR():
    
    def __init__(self, N=1, *prob_params,  **init_cond):
        
        # Set the default arguments for SIR initial conditions   
        self.N  = N 
        self.I0 = 1e-6*self.N # 1 in million infections
        self.R0 = 0           # No initial immunity

        #If inputs provided, override the default values
        for key in init_cond:
            setattr(self, key, init_cond[key])

        # Set default parameter distributions
        if not prob_params:
            sim_init_cond       = loadSimulationParams(5, scenario = 0, plot_data = 0)
            prob_params, _      = getSIRTestingParams(3, 'gamma',**sim_init_cond)

        # Set input parameter distributions
        self.beta_dist, self.gamma_inv_dist = set_distribution_params(*prob_params)

    
    def deriv(self, beta, gamma):
        def ddt(y,t):
            S, I, R = y
            dSdt = -beta/self.N * S * I
            dIdt = beta/self.N * S * I  - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
        return ddt

    def sample_params(self):
        
        samples = {}
        
        if isinstance(self.beta_dist, stats._distn_infrastructure.rv_frozen):
            samples['beta'] = self.beta_dist.rvs(1)[0]
        else:
            samples['beta'] = self.beta_dist.rvs(1)[0]            

        if isinstance(self.gamma_inv_dist, stats._distn_infrastructure.rv_frozen):
            samples['gamma'] = 1/self.gamma_inv_dist.rvs(1)[0]
        else:
            samples['gamma'] = 1/self.gamma_inv_dist.rvs(1)[0]

        return samples
    
    def rollout(self, days, dt = 1):
        
        samples = self.sample_params()
        print(samples)
        
        #With the samples and the initial conditions for the models, rollout the IVP
        deriv_fun = self.deriv(samples['beta'], samples['gamma'])
        S0 = self.N - self.I0 - self.R0
        y0 = (S0, self.I0, self.R0)
        t = np.arange(0, days)
        
        ode_sol  = odeint(deriv_fun, y0, t)
        S,I,R    = ode_sol.T
        
        return (S,I,R), samples
    
    def project(self, days, samples = 10000, dt = 1, progbar = True):
        
        t = np.arange(0, days)
        S_samples = np.empty((samples, days))
        I_samples = np.empty((samples, days))
        R_samples = np.empty((samples, days))
        param_samples = []
        
        for i in (tqdm(range(samples)) if progbar else range(samples)):
            (S,I,R), sample = self.rollout(days, dt)
            S_samples[i,:] = S
            I_samples[i,:] = I
            R_samples[i,:] = R
            param_samples.append(sample)
        
        return (S_samples, I_samples, R_samples), param_samples

if __name__ == '__main__':
    
    model = StochasticSIR()
    traces, samples = model.project(days = 200, samples = 10000, progbar = False)
        