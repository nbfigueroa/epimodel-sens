# -*- coding: utf-8 -*-
from utils import *
import scipy.stats as stats
from sims import *

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
        self.sigma_dist = stats.gamma(a = sigma_inv_shape, loc = sigma_inv_loc, scale = sigma_inv_scale)
        
        self.I0 = 1e-6*self.N #1 in million infections
        
        #If inputs provided, override the default values
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
    
    
    @property
    def E0(self):
        return 10*self.I0


if __name__ == '__main__':
    
    model = StochasticSEIR()