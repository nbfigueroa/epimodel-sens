# -*- coding: utf-8 -*-
from epimodels.utils import *
from epimodels.sims import *
import scipy.stats as stats
from scipy.integrate  import odeint
#from sims import *
import numpy as np
from tqdm import tqdm

class StochasticSEIR():
    
    def __init__(self, N=1, *prob_params,  **init_cond):
        
        #Set the default arguments
        
        #Load gamma distribution params from sims
        self.N = N #The default value only computes proportional prevalence
        self.I0 = 1e-6*self.N # 1 in million infections
        self.E0 = 10*self.I0
        self.R0 = 0           # No initial immunity
        # Load the default distribution parameters
        
        if not prob_params:
            sim_init_cond       = loadSimulationParams(5, scenario = 0, plot_data = 0)
            prob_params, _      = getSIRTestingParams(3, 'gamma',**sim_init_cond)
        
        
        self.beta_dist, self.gamma_inv_dist, self.sigma_inv_dist = self.set_distribution_params(*prob_params)
        # self.I0 = 1e-6*self.N #1 in million infections
        # self.E0 = 10*self.I0
        # self.R0 = 0 #No initial immunity
        # self.text_error = text_error
        # self._ext = _ext
        
        #If inputs provided, override the default values
        for key in init_cond:
            setattr(self, key, init_cond[key])
        
    
    def set_distribution_params(*prob_params):

        # Sample from Gamma Distributions        
        if prob_params[1] == 'gamma':   
            beta_loc        = prob_params[2]
            beta_shape      = prob_params[3]
            beta_scale      = prob_params[4]
            gamma_inv_loc   = prob_params[5]
            gamma_inv_shape = prob_params[6]
            gamma_inv_scale = prob_params[7]
            sigma_inv_loc   = prob_params[8]
            sigma_inv_shape = prob_params[9]
            sigma_inv_scale = prob_params[10]
            
            if beta_scale == 0:
                beta_dist      = stats.norm(loc = beta_loc, scale = beta_scale)
            else:        
                beta_dist      = stats.gamma(a = beta_shape, loc = beta_loc, scale = beta_scale)
            
            if gamma_inv_scale == 0:
                gamma_inv_dist = stats.norm(loc = gamma_inv_loc, scale = gamma_inv_scale)
            else:    
                gamma_inv_dist = stats.gamma(a = gamma_inv_shape, loc = gamma_inv_loc, scale = gamma_inv_scale)
            
            if sigma_inv_scale == 0:
                sigma_inv_dist = stats.norm(loc = sigma_inv_loc, scale = sigma_inv_scale)
            else:
                sigma_inv_dist = stats.gamma(a = sigma_inv_shape, loc = sigma_inv_loc, scale = sigma_inv_scale)
            
        # Sample from LogNormal Distributions        
        if prob_params[1] == 'log-normal':
            beta_mean       = prob_params[2]
            beta_std        = prob_params[3]
            gamma_inv_mean  = prob_params[4]
            gamma_inv_std   = prob_params[5]
            sigma_inv_mean  = prob_params[6]
            sigma_inv_std   = prob_params[7]


            if beta_std == 0:
                beta_dist  = stats.norm(loc = beta_mean, scale = beta_std)
            else:  
                beta_dist  = stats.lognorm(scale = np.exp(beta_mean),s = beta_std)

            # Sample from LogNormal if std not 0
            if gamma_inv_std == 0:
                gamma_inv_dist = stats.norm(loc = gamma_inv_mean, scale = gamma_inv_std)
            else:  
                gamma_inv_dist = stats.lognorm(scale = np.exp(gamma_inv_mean), s = gamma_inv_std)
            
            if sigma_inv_std == 0:
                sigma_inv_dist = stats.norm(loc = sigma_inv_mean, scale = sigma_inv_std)
            else:
                sigma_inv_dist = stats.lognorm(scale = np.exp(sigma_inv_mean), s = sigma_inv_std)


        # Sample from Gaussian Distributions        
        if prob_params[1] == 'gaussian':
            beta_mean       = prob_params[2]
            beta_std        = prob_params[3]
            gamma_inv_mean  = prob_params[4]
            gamma_inv_std   = prob_params[5]
            sigma_inv_mean  = prob_params[6]
            sigma_inv_std   = prob_params[7]
            
            # Sample from Gaussian Distributions
            beta_dist       = stats.norm(loc = beta_mean, scale = beta_std)
            gamma_inv_dist  = stats.norm(loc = gamma_inv_mean, scale = gamma_inv_std)
            sigma_inv_dist  = stats.norm(loc = sigma_inv_mean, scale = sigma_inv_std)
            


        # Sample from Uniform Distributions        
        if prob_params[1] == 'uniform':        
            if prob_params[2] == prob_params[3]:
                beta_dist = stats.norm(loc = prob_params[2], scale = 0)
            else:
                beta_dist = stats.uniform(loc = prob_params[2], scale = prob_params[3] - prob_params[2])

            if prob_params[4] == prob_params[5]:
                gamma_inv_dist = stats.norm(loc = prob_params[4], scale = 0)
            else:
                gamma_inv_dist = stats.uniform(loc = prob_params[4], scale = prob_params[5] - prob_params[4])
            
            if prob_params[6] == prob_params[7]:
                sigma_inv_dist = stats.norm(loc = prob_params[6], scale = 0)
            else:
                sigma_inv_dist = stats.uniform(loc = prob_params[6], scale = prob_params[7] - prob_params[6])


        return beta_dist, gamma_inv_dist, sigma_inv_dist
    
    
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
        
        # These hacks are necessary for the gaussian distribution, which is wrong to use anyway
        if isinstance(self.beta_dist, stats._distn_infrastructure.rv_frozen):
            beta = self.beta_dist.rvs(1)[0]
            while beta < 0.15:
                beta = self.beta_dist.rvs(1)[0]

            samples['beta'] = beta
        else:
            samples['beta'] = self.beta_dist

        if isinstance(self.gamma_inv_dist, stats._distn_infrastructure.rv_frozen):
            gamma_inv = self.gamma_inv_dist.rvs(1)[0]
            while gamma_inv < 3:
                gamma_inv = self.gamma_inv_dist.rvs(1)[0]

            samples['gamma'] = 1/gamma_inv
        else:
            samples['gamma'] = 1/self.gamma_inv_dist
        
        if isinstance(self.sigma_inv_dist, stats._distn_infrastructure.rv_frozen):
            samples['sigma'] = 1/self.sigma_inv_dist.rvs(1)[0]
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
        
    
    def project(self, days, samples = 10000, dt = 1, progbar = True):
        
        t = np.arange(0, days)
        S_samples = np.empty((samples, days))
        E_samples = np.empty((samples, days))
        I_samples = np.empty((samples, days))
        R_samples = np.empty((samples, days))
        param_samples = np.empty((samples, 3))
        
        for i in (tqdm(range(samples)) if progbar else range(samples)):
            (S,E,I,R), sample = self.rollout(days, dt)
            S_samples[i,:] = S
            E_samples[i,:] = E
            I_samples[i,:] = I
            R_samples[i,:] = R
            param_samples[i,:]= [sample['beta'],1/sample['gamma'], 1/sample['sigma']]
        
        return (S_samples, E_samples, I_samples, R_samples), param_samples




if __name__ == '__main__':
    
    sim_kwargs = loadSimulationParams(5, 0, plot_data = 0, header = 'SEIR')
    prob_params, plot_vars = getSEIRTestingParams(1, 'gamma',**sim_kwargs)
    init_cond = {}
    init_cond['I0'] = sim_kwargs['I0']
    init_cond['R0'] = sim_kwargs['R0']
    init_cond['E0'] = sim_kwargs['E0']
    model = StochasticSEIR(sim_kwargs['N'], *prob_params, **init_cond)
    
    