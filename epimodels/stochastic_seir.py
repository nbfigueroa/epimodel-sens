# -*- coding: utf-8 -*-
from utils import *
import scipy.stats as stats
from scipy.integrate  import odeint
from sims import *
import numpy as np

class StochasticSEIR():
    
    def __init__(self, N=1,  **kwargs):
        
        #Set the default arguments
        
        #Load gamma distribution params from sims
        #text_error, prob_params, _ext = getSIRTestingParams(3, prob_type,**sim_kwargs)
        self.N = N #The default value only computes proportional prevalence
        
        # Load the default distribution parameters
        sim_kwargs             = loadSimulationParams(5, 0, plot_data = 0)
        text_error, prob_params, _ext = getSIRTestingParams(3, 'gamma',**sim_kwargs)
        
        # Generate the default distributions for beta
        beta_loc        = prob_params[1]
        beta_scale      = prob_params[2]
        beta_shape      = prob_params[3]
        gamma_inv_loc   = prob_params[4]
        gamma_inv_scale = prob_params[5]
        gamma_inv_shape = prob_params[6]
        sigma_inv_loc   = 2.2
        sigma_inv_shape = 3.35
        sigma_inv_scale = 0.865
        
        self.beta_dist = stats.gamma(a = beta_shape, loc = beta_loc, scale = beta_scale)
        self.gamma_inv_dist = stats.gamma(a = gamma_inv_shape, loc = gamma_inv_loc, scale = gamma_inv_scale)
        self.sigma_inv_dist = stats.gamma(a = sigma_inv_shape, loc = sigma_inv_loc, scale = sigma_inv_scale)
        
        self.I0 = 1e-6*self.N #1 in million infections
        self.E0 = 10*self.I0
        self.R0 = 0 #No initial immunity
        self.text_error = text_error
        self._ext = _ext
        
        #If inputs provided, override the default values
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
    
    
    # @property
    # def E0(self):
    #     return 10*self.I0
    
    def deriv(self, beta, sigma, gamma):
        
        def ddt(y,t):
            S, E, I, R = y
            N = self.N
            # Main state variables with exponential rates
            dSdt = -(beta * I * S)/N 
            dEdt =  (beta*S*I)/N - sigma*E
            dIdt  = sigma*E - gamma*I
            dRdt = gamma * I
        
            return dSdt, dEdt, dIdt, dRdt
        return ddt
    
    def sample_params(self):
        
        samples = {}
        
        if isinstance(self.beta_dist, stats._distn_infrastructure.rv_frozen):
            samples['beta'] = self.beta_dist.rvs()
        else:
            samples['beta'] = self.beta_dist
        
        if isinstance(self.gamma_inv_dist, stats._distn_infrastructure.rv_frozen):
            samples['gamma'] = 1/self.gamma_inv_dist.rvs()
        else:
            samples['gamma'] = 1/self.gamma_inv_dist
        
        if isinstance(self.sigma_inv_dist, stats._distn_infrastructure.rv_frozen):
            samples['sigma'] = 1/self.sigma_inv_dist.rvs()
        else:
            samples['sigma'] = 1/self.sigma_inv_dist
        
        return samples
        
    
    def rollout(self, days, dt = 1):
        
        samples = self.sample_params()
        
        #With the samples and the initial conditions for the models, rollout the IVP
        deriv_fun = self.deriv(samples['beta'], samples['gamma'], samples['sigma'])
        S0 = self.N - self.I0 - self.E0 - self.R0
        y0 = (S0, self.E0, self.I0, self.R0)
        t = np.arange(0, days)
        
        ode_sol  = odeint(deriv_fun, y0, t)
        S,E,I,R    = ode_sol.T
        
        return (S,E,I,R), samples
        
    
    def project(self, days, samples = 10000, dt = 1):
        
        t = np.arange(0, days)
        S_samples = np.empty((samples, days))
        E_samples = np.empty((samples, days))
        I_samples = np.empty((samples, days))
        R_samples = np.empty((samples, days))
        param_samples = []
        
        for i in range(samples):
            (S,E,I,R), sample = self.rollout(days, dt)
            S_samples[i,:] = S
            E_samples[i,:] = E
            I_samples[i,:] = I
            R_samples[i,:] = R
            param_samples.append(sample)
        
        return (S_samples, E_samples, I_samples, R_samples), param_samples




if __name__ == '__main__':
    
    model = StochasticSEIR()
    traces, samples = model.project(30)