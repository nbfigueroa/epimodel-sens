import numpy             as np
from   scipy             import stats
from   scipy.stats       import gamma as gamma_dist
import matplotlib.pyplot as plt
from tqdm                import tqdm

# Custom classes and helper/plotting functions
from epimodels.sir    import *
from epimodels.utils  import *
from epimodels.plots  import *
from epimodels.sims   import *


### TODO: Add the growth rate plots with errors

## TODO: Put this in new stoche-SIR [stochastic estimate] class
def rollout_SIR_sim_stoch(*prob_params, **kwargs):
    '''
        Run a single simulation of stochastic SIR dynamics
    '''
    verbose  = kwargs['verbose']
    N        = kwargs['N']
    days     = kwargs['days']

    # Sample from Uniform Distributions        
    if prob_params[0] == 'uniform':        
        if prob_params[1] == prob_params[2]:
            beta = prob_params[1]
        else:
            beta = np.random.uniform(prob_params[1],prob_params[2])
        if prob_params[3] == prob_params[4]:
            gamma_inv = prob_params[3]
        else:
            gamma_inv = np.random.uniform(prob_params[3],prob_params[4])

    # Sample from Gaussian Distributions        
    if prob_params[0] == 'gaussian':        
        beta_mean       = prob_params[1]
        beta_std        = prob_params[2]
        gamma_inv_mean  = prob_params[3]
        gamma_inv_std   = prob_params[4]
        
        # Sample from Gaussian Distributions
        beta = 0
        while beta < 0.02:
            beta            = np.random.normal(beta_mean, beta_std)
        gamma_inv = 0    
        while gamma_inv < 0.1:
            gamma_inv       = np.random.normal(gamma_inv_mean, gamma_inv_std)

    # Sample from Gamma Distributions        
    if prob_params[0] == 'gamma':        
        beta_loc        = prob_params[1]
        beta_shape      = prob_params[2]
        beta_scale      = prob_params[3]
        gamma_inv_loc   = prob_params[4]
        gamma_inv_shape = prob_params[5]
        gamma_inv_scale = prob_params[6]

        # Sample from Gamma Distributions
        if beta_scale == 0:
            beta = beta_loc
        else:        
            beta_dist      = gamma_dist(beta_shape, beta_loc, beta_scale)
            beta           = beta_dist.rvs(1)[0]
        if gamma_inv_scale == 0:
            gamma_inv = gamma_inv_loc
        else:            
            gamma_inv_dist = gamma_dist(gamma_inv_shape, gamma_inv_loc, gamma_inv_scale)
            gamma_inv      = gamma_inv_dist.rvs(1)[0]

    # Sample from LogNormal Distributions        
    if prob_params[0] == 'log-normal':
        beta_mean       = prob_params[1]
        beta_std        = prob_params[2]
        gamma_inv_mean  = prob_params[3]
        gamma_inv_std   = prob_params[4]

        # Sample from LogNormal if std not 0
        if beta_std == 0:
            beta = beta_mean
        else:  
            beta  = np.random.lognormal(beta_mean,beta_std)

        # Sample from LogNormal if std not 0
        if gamma_inv_std == 0:
            gamma_inv = gamma_inv_mean
        else:  
            gamma_inv = np.random.lognormal(gamma_inv_mean, gamma_inv_std)

    # Derived values    
    gamma      = 1.0 / gamma_inv
    r0         = beta * gamma_inv

    if verbose:
        print('*****   SIMULATING SIR MODEL DYNAMICS *****')    
        print('*****         Hyper-parameters        *****')
        print('N=',N,'days=', days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)
        print('*****         Model-parameters        *****')
        print('beta=',beta, 'gamma=',gamma)

    # Create Model
    model_kwargs = {}
    model_kwargs['r0']         = r0
    model_kwargs['inf_period'] = gamma_inv
    model_kwargs['I0']         = kwargs['I0']
    model_kwargs['R0']         = kwargs['R0']

    model   = SIR(N,**model_kwargs)
    S,I,R,t = model.project(days,'ode_int')

    return S, I, R, t, beta, gamma_inv

## TODO: Put this in new stoche-SIR [stochastic estimate] class
def mc_SIR_sim_stoch(text_error, rollouts, viz_plots= 0, *prob_params, **kwargs):
    '''
        Run Monte Carlo Simulations of the Stochastic Estimate of SIR
    '''
    S_samples          = np.empty([rollouts, kwargs['days']])
    I_samples          = np.empty([rollouts, kwargs['days']])
    R_samples          = np.empty([rollouts, kwargs['days']])
    beta_samples       = np.empty([rollouts, 1])
    gamma_inv_samples  = np.empty([rollouts, 1])

    ############################################################################
    ######## Simulate Vanilla SIR Model Dynamics for each value of beta ########
    ############################################################################
    for ii in tqdm(range(rollouts)):
        S, I, R, t, beta, gamma_inv   = rollout_SIR_sim_stoch(*prob_params, **kwargs)

        # Storing run in matrix for post-processing
        S_samples[ii,:]         = S
        I_samples[ii,:]         = I
        R_samples[ii,:]         = R
        beta_samples[ii,:]      = beta
        gamma_inv_samples[ii,:] = gamma_inv

    # Compute stats from MC rollouts
    S_stats, I_stats, R_stats, T_stats = gatherMCstats(S_samples, I_samples, R_samples, bound_type = 'Quantiles')
    
    show_results   = 1
    if show_results:
        print('*********   MEAN Results    *********')    
        I_mean = I_stats[0,:]
        R_mean = R_stats[0,:]
        T_mean = I_mean  + R_mean
        tc, t_I100, t_I500, t_I100, t_I10 = getCriticalPointsAfterPeak(I_mean)
        T_tc  = T_mean[tc]
        print('Total Cases @ Peak = ', T_tc,'by day=', tc)
        total_infected     = I_mean[-1]
        print('Infected @ t(end) = ', total_infected)
        total_cases     = T_mean[-1]
        print('Total Cases @ t(end) = ', total_cases)

    ##############################################################
    ######## Plots Simulation Variables with Error Bounds ########
    ##############################################################
    # Plot Realizations of Infected and Total Cases
    plotIT_realizations(I_samples, R_samples, **kwargs)

    # Plot of Critical Points on realizations
    plotCriticalPointsStats(I_samples, R_samples, **kwargs)

    # Plot Histogram of Sampled Parameters beta and gamma
    filename = kwargs['file_extension'] + "_paramSamples"
    plotSIR_sampledParams(beta_samples, gamma_inv_samples, filename,  *prob_params)

    print('Beta interval: [', beta_samples[np.argmin(beta_samples)], ',', beta_samples[np.argmax(beta_samples)],']')
    print('Gamma^-1 interval: [', gamma_inv_samples[np.argmin(gamma_inv_samples)], ',', gamma_inv_samples[np.argmax(gamma_inv_samples)],']')

    # Plot SIR Curves with expected values, CI and standard deviation
    x_axis_offset       = round(kwargs['days']*0.25)
    y_axis_offset       = 0.0000003 
    plot_all            = 1; plot_peaks = 1; show_S = 0; show_T = 1; show_R = 0; show_analytic_limit = 0; scale_offset = 0.01 
    Plotoptions         = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset
    plotSIR_evolutionStochastic(S_stats, I_stats, R_stats, T_stats, Plotoptions, text_error, **kwargs)    

    if viz_plots:
        plt.show()


def main():    

    # Choose Simulation
    '''Simulation options defined in sims.py
        Defines different initial condition and base parameters 
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Primer case study
    '''
    sim_num                = 5
    sim_kwargs             = loadSimulationParams(sim_num, 0, plot_data = 0)
    sim_kwargs['verbose']  = 0    
    basefilename           = sim_kwargs['file_extension']
    
    # Testing Variants
    '''Testing options defined in sims.py
        test_num = 1  --> Sample beta, gamma fixed
        test_num = 2  --> fix beta, sample gamma
        test_num = 3  --> sample beta and gamma
    '''

    # Probability Distribution type
    '''Parameters for each probability distribution type defined in sims.py
        uniform   --> lb, ub
        gaussian  --> mean, std
        gamma     --> loc, shape (k), scale (theta)        
        log-Normal --> mean, std
    '''

    prob_type = 'gamma'    
    rollouts  = pow(10,2)  
    viz_plots = 1 
    
    for test_num in range(3):
        if test_num == 3:
            sim_kwargs['do_paramCountours'] = 1
        else:
            sim_kwargs['do_paramCountours'] = 0
        text_error, prob_params, _ext = getSIRTestingParams(test_num+1, prob_type,**sim_kwargs)
        sim_kwargs['file_extension']  = basefilename + _ext
        
        # Run Montecarlo simulation for chosen parameter test
        mc_SIR_sim_stoch(text_error, rollouts, viz_plots, *prob_params, **sim_kwargs)
    
if __name__ == '__main__':
    main()