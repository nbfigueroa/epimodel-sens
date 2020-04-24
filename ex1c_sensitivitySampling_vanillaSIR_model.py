import numpy             as np
import matplotlib.pyplot as plt
from tqdm                import tqdm

# Custom classes and helper/plotting functions
from epimodels.stochastic_sir    import *
from epimodels.sir    import *
from epimodels.utils  import *
from epimodels.plots  import *
from epimodels.sims   import *



### TODO: Add the growth rate plots with errors
def run_MC_SIRparams(rollouts, *prob_params, **sim_kwargs):
    '''
        Run Monte Carlo Simulations of the Stochastic Estimate of SIR
    '''

    #################################################################################
    ####### Generate Predictions for Sampled Values of \beta and \gamma^{-1} ########
    #################################################################################

    # Run MC rollouts
    init_cond = {}
    init_cond['I0'] = sim_kwargs['I0']
    init_cond['R0'] = sim_kwargs['R0']
    model = StochasticSIR(sim_kwargs['N'], *prob_params, **init_cond)
    SIR_traces, SIR_params = model.project(days = sim_kwargs['days'], samples = rollouts, progbar = True)


    ##############################################################
    ######## Plots and Results Statistics and Realizations #######
    ##############################################################
    # Unpack rollout traces
    S_samples, I_samples, R_samples = SIR_traces
    beta_samples      = SIR_params[:,0]
    gamma_inv_samples = SIR_params[:,1]

    # Plot Histogram of Sampled Parameters beta and gamma    
    filename          = sim_kwargs['file_extension'] + "_paramSamples"
    plotSIR_sampledParams(beta_samples, gamma_inv_samples, filename,  *prob_params)
    print('Beta interval: [', beta_samples[np.argmin(beta_samples)], ',', beta_samples[np.argmax(beta_samples)],']')
    print('Gamma^-1 interval: [', gamma_inv_samples[np.argmin(gamma_inv_samples)], ',', gamma_inv_samples[np.argmax(gamma_inv_samples)],']')
    
    # Plot Realizations of Infected and Total Cases
    plotIT_realizations(I_samples, R_samples, **sim_kwargs)

    # Plot of Critical Points on realizations
    plotCriticalPointsStats(I_samples, R_samples, **sim_kwargs)

    # Plot SIR Curves with expected values, CI and standard deviation
    S_stats, I_stats, R_stats, T_stats = gatherMCstats(S_samples, I_samples, R_samples, bound_type = 'Quantiles')    
    printMeanResults(I_stats, R_stats)

    x_axis_offset       = round(sim_kwargs['days']*0.25)
    y_axis_offset       = 0.0000003 
    plot_all            = 1; plot_peaks = 1; show_S = 0; show_T = 1; show_R = 0; show_analytic_limit = 0; scale_offset = 0.01 
    Plotoptions         = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset
    plotSIR_evolutionStochastic(S_stats, I_stats, R_stats, T_stats, Plotoptions, **sim_kwargs)    

    if sim_kwargs['viz_plots']:
        plt.show()


def main():

    # Choose Simulation (includes initial conditions, fixed model parameters and plotting options)
    '''Simulation parameters defined in sims.py
        Defines different initial condition and base parameters 
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Primer case study
    '''
    sim_num                = 5
    sim_kwargs             = loadSimulationParams(sim_num, 0, plot_data = 0)
    basefilename           = sim_kwargs['file_extension']
    
    # Testing Variants (includes test type and probability distribution type)
    '''Testing options defined in sims.py
        test_num = 1  --> Sample beta, gamma fixed
        test_num = 2  --> fix beta, sample gamma
        test_num = 3  --> sample beta and gamma
       
       Parameters for each probability distribution type defined in sims.py
        uniform   --> lb, ub
        gaussian  --> mean, std
        gamma     --> loc, shape (k), scale (theta)        
        log-Normal --> mean, std
    '''

    prob_type = 'gamma'    
    rollouts  = pow(10,3)  
    viz_plots = 1 
    
    for test_num in range(3):
        prob_params, plot_vars        = getSIRTestingParams(test_num+1, prob_type,**sim_kwargs)
        
        # unpack plotting and file variables
        text_error, _ext              = plot_vars
        sim_kwargs['file_extension']  = basefilename + _ext
        sim_kwargs['text_error']      = text_error
        sim_kwargs['viz_plots']       = viz_plots

        # Run Montecarlo simulation for chosen parameter test
        run_MC_SIRparams(rollouts, *prob_params, **sim_kwargs)
    
if __name__ == '__main__':
    main()