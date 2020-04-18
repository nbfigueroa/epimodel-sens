import math
import numpy             as np
from   scipy             import stats
from   scipy.stats       import gamma as gamma_dist
import matplotlib.pyplot as plt
from   matplotlib        import rc
from tqdm                import tqdm

# Custom classes and helper/plotting functions
from epimodels.sir   import *
from epimodels.utils import *
from epimodels.sims  import *

### TODO: Add the growth rate plots with errors
def computeStats(X):
    '''
        Compute mean, median and confidence intervals
    '''
    X_bar = np.mean(X, axis=0) # mean of vector
    X_std = np.std(X, axis=0) # std of vector
    n     = len(X_bar) # number of obs
    z     = 1.96 # for a 95% CI
    X_lower = X_bar - (z * (X_std/math.sqrt(n)))
    X_upper = X_bar + (z * (X_std/math.sqrt(n)))
    X_med   = np.median(X, axis=0)        
    return X_bar, X_med, X_std, X_upper, X_lower

def gatherMCstats(S_samples, I_samples, R_samples):    
    '''
        Gather stats from MC simulations  
    '''
    S_mean, S_med, S_std, S_upper, S_lower = computeStats(S_samples)
    I_mean, I_med, I_std, I_upper, I_lower = computeStats(I_samples)
    R_mean, R_med, R_std, R_upper, R_lower = computeStats(R_samples)

    # Pack values for plotting and analysis
    S_stats          = np.vstack((S_mean, S_upper, S_lower, S_std))    
    I_stats          = np.vstack((I_mean, I_upper, I_lower, I_std))    
    R_stats          = np.vstack((R_mean, R_upper, R_lower, R_std))    
    return S_stats, I_stats, R_stats

## TODO: Move to utils.py or to new stoche-SIR [stochastic estimate] class
def plotSIR_sampledParams(beta_samples, gamma_inv_samples, filename, *prob_params):
    fig, (ax1,ax2) = plt.subplots(1,2, constrained_layout=True)

    ###########################################################
    ################## Plot for Beta Samples ##################
    ###########################################################
    count, bins, ignored = ax1.hist(beta_samples, 30, density=True)

    if prob_params[0] == 'uniform':
        ax1.set_xlabel(r"$\beta \sim \mathcal{N}$", fontsize=15)        

    if prob_params[0] == 'gaussian':
        mu    = prob_params[1]
        sigma = prob_params[2] + 0.00001
        ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        ax1.set_xlabel(r"$\beta \sim \mathcal{N}$", fontsize=15)    

    if prob_params[0] == 'gamma':
        g_dist    = gamma_dist(prob_params[2], prob_params[1], prob_params[3])
        # Plot gamma samples and pdf
        x = np.arange(0,1,0.001)
        ax1.plot(x, g_dist.pdf(x), 'r',label=r'$k = 1, \mu=%.1f,\ \theta=%.1f$' % (prob_params[1], prob_params[2]))


    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
    plt.xlim(0, 1.0)            
    ax1.grid(True, alpha=0.3)
    ax1.set_title(r"Histogram of $\beta$ samples", fontsize=20)
    
    ###############################################################
    ################## Plot for Gamma^-1 Samples ##################
    ###############################################################
    count, bins, ignored = ax2.hist(gamma_inv_samples, 30, density=True)
    if prob_params[0] == 'gaussian':
        mu    = prob_params[3]
        sigma = prob_params[4] + 0.00001
        ax2.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')

        ax2.set_xlabel(r"$\gamma^{-1} \sim \mathcal{N}$", fontsize=15)
    
    if prob_params[0] == 'uniform':
        ax2.set_xlabel(r"$\gamma^{-1} \sim \mathcal{U}$", fontsize=15)          

    if prob_params[0] == 'gamma':
        g_dist    = gamma_dist(prob_params[5], prob_params[4], prob_params[6])
        # Plot gamma samples and pdf
        x = np.arange(1,15,0.1)
        ax2.plot(x, g_dist.pdf(x), 'r',label=r'$k = 1, \mu=%.1f,\ \theta=%.1f$' % (prob_params[3], prob_params[4]))


    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)  
    plt.xlim(1, 17) 
    ax2.grid(True, alpha=0.3)    
    plt.title(r"Histogram of $\gamma^{-1}$ samples", fontsize=20)    

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(20/2, 8/2, forward=True)    
    
    # Store plot
    plt.savefig(filename + ".png", bbox_inches='tight')

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

    # Sample from Gaussian Distributions        
    if prob_params[0] == 'gamma':        
        beta_loc        = prob_params[1]
        beta_scale      = prob_params[2]
        beta_shape      = prob_params[3]
        gamma_inv_loc   = prob_params[4]
        gamma_inv_scale = prob_params[5]
        gamma_inv_shape = prob_params[6]

        # Sample from Gamma Distributions
        if beta_scale == 0:
            beta = beta_loc
        else:        
            beta_dist      = gamma_dist(beta_scale, beta_loc, beta_shape)
            beta           = beta_dist.rvs(1)[0]
        if gamma_inv_scale == 0:
            gamma_inv = gamma_inv_loc
        else:            
            gamma_inv_dist = gamma_dist(gamma_inv_scale, gamma_inv_loc, gamma_inv_shape)
            gamma_inv      = gamma_inv_dist.rvs(1)[0]

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
def mc_SIR_sim_stoch(text_error, rollouts, *prob_params, **kwargs):
    '''
        Run Monte Carlo Simulations of the Stochastic SIR
    '''
    show_results   = 1

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
    S_stats, I_stats, R_stats = gatherMCstats(S_samples, I_samples, R_samples)
    
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
    # Plot Realizations of Infected and Recovered
    plotIR_realizations(I_samples, R_samples, **kwargs)

    # Plot Histogram of Sampled Parameters beta and gamma
    filename = kwargs['file_extension'] + "_paramSamples"
    plotSIR_sampledParams(beta_samples, gamma_inv_samples, filename,  *prob_params)

    # Plot SIR Curves with expected values, CI and standard deviation
    x_axis_offset       = round(kwargs['days']*0.25)
    y_axis_offset       = 0.0000003 
    plot_all            = 1; plot_peaks = 1; show_S = 1; show_T = 1; show_R = 0; show_analytic_limit = 0; scale_offset = 0.01 
    Plotoptions         = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset
    plotSIR_evolutionStochastic(S_stats, I_stats, R_stats, Plotoptions, text_error, **kwargs)    

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
    
    # Choose Testing Variant
    '''Testing options defined in sims.py
        test_num = 1  --> Sample beta, gamma fixed
        test_num = 2  --> fix beta, sample gamma
        test_num = 3  --> sample beta and gamma
    '''
    prob_type = 'gamma'    
    rollouts  = pow(10,4)   
    
    for test_num in range(3):
        text_error, prob_params, _ext = getSIRTestingParams(test_num+1, prob_type,**sim_kwargs)
        sim_kwargs['file_extension']  = basefilename + _ext
        
        # Run Montecarlo simulation for chosen parameter test
        mc_SIR_sim_stoch(text_error, rollouts, *prob_params, **sim_kwargs)
        # plt.show()
    
if __name__ == '__main__':
    main()