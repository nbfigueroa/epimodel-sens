# -*- coding: utf-8 -*-
from epimodels.stochastic_seir    import *
from epimodels.utils  import *
from epimodels.plots  import *
from epimodels.sims   import *


def computeSEIR_MCresults(SEIR_traces, SEIR_params, *prob_params, **sim_kwargs):
    '''
        Compute and Plot results from Monte Carlo simulation of SIR-type models
    '''

    # Unpack rollout traces
    S_samples, E_samples, I_samples, R_samples = SEIR_traces
    beta_samples      = SEIR_params[:,0]
    gamma_inv_samples = SEIR_params[:,1]
    sigma_inv_samples = SEIR_params[:,2]

    # Plot Histogram of Sampled Parameters beta and gamma    
    filename          = sim_kwargs['file_extension'] + "_paramSamples"
    plotSEIR_sampledParams(beta_samples, gamma_inv_samples, sigma_inv_samples, filename,  *prob_params)
    print('Beta interval: [', beta_samples[np.argmin(beta_samples)], ',', beta_samples[np.argmax(beta_samples)],']')
    print('Gamma^-1 interval: [', gamma_inv_samples[np.argmin(gamma_inv_samples)], ',', gamma_inv_samples[np.argmax(gamma_inv_samples)],']')
    
    # Critical outputs from realizations
    CO_samples = getCriticalPointsDistribution(I_samples, R_samples)
    # Variables: tc_samples, Ipeak_samples, Tend_samples = CO_samples
    plot_data_quant  = 0; plot_regress_lines = 0; do_histograms = 1; do_contours = 1; do_mask = 1
    plot_options = plot_data_quant, plot_regress_lines, do_histograms, do_contours, do_mask
    computeCriticalPointsStats(SEIR_params, CO_samples, plot_options, **sim_kwargs)

    plot_traces = 1
    if plot_traces:
        ### TODO: Add the growth rate plots with errors    
        # plotSIR_growth_realizations(SIR_traces, SIR_params)
        
        # Plot Realizations of Infected and Total Cases
        plotIT_realizations(I_samples, R_samples, **sim_kwargs)

        # Plot SIR Curves with expected values, CI and standard deviation
        S_stats, E_stats, I_stats, R_stats, T_stats = gatherMCstats(S_samples, I_samples, R_samples, E_samples = E_samples, bound_type = 'Quantiles', bound_param = [0.025, 0.975])    
        printMeanResults(I_stats, R_stats)

        x_axis_offset       = round(sim_kwargs['days']*0.25)
        y_axis_offset       = 0.0000003 
        plot_all            = 1; plot_peaks = 1; show_S = 0; show_E = 1; show_T = 1; show_R = 0; show_analytic_limit = 0; scale_offset = 0.01 
        Plotoptions         = plot_all, show_S, show_E, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset
        plotSEIR_evolutionStochastic(S_stats, E_stats, I_stats, R_stats, T_stats, Plotoptions, **sim_kwargs)    

    if sim_kwargs['viz_plots']:
        plt.show()


def run_MC_stochastic_est_SEIR(rollouts, *prob_params, **sim_kwargs):
    '''
        Run Monte Carlo Simulations of the Stochastic Estimates of SIR traces
    '''

    #################################################################################
    ####### Generate Predictions for Sampled Values of \beta and \gamma^{-1} ########
    #################################################################################

    # Run MC rollouts
    init_cond = {}
    init_cond['I0'] = sim_kwargs['I0']
    init_cond['R0'] = sim_kwargs['R0']
    init_cond['E0'] = sim_kwargs['E0']
    model = StochasticSEIR(sim_kwargs['N'], *prob_params, **init_cond)
    SEIR_traces, SEIR_params = model.project(days = sim_kwargs['days'], samples = rollouts, progbar = True)


    ##############################################################
    ######## Plots and Results Statistics and Realizations #######
    ##############################################################
    computeSEIR_MCresults(SEIR_traces, SEIR_params, *prob_params, **sim_kwargs)


def run(prob_type = 'gamma', days = 200):

    # Choose Simulation (includes initial conditions, fixed model parameters and plotting options)
    '''Simulation parameters defined in sims.py
        Defines different initial condition and base parameters 
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Primer case study
    '''
    sim_num             = 5
    sim_kwargs          = loadSimulationParams(sim_num, 0, plot_data = 0, header = 'SEIR')
    sim_kwargs['days']  = days
    basefilename        = sim_kwargs['file_extension']
    workbook, worksheet = createResultsfile(basefilename, prob_type, test_type='sampling')

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
    rollouts  = pow(10,5)
    viz_plots = 0

    for test_num in [1, 2, 3, 4, 5]:
        prob_params, plot_vars        = getSEIRTestingParams(test_num=test_num, prob_type=prob_type,**sim_kwargs)
        
        # unpack plotting and file variables
        text_error, _ext              = plot_vars
        sim_kwargs['file_extension']  = basefilename + _ext
        sim_kwargs['text_error']      = text_error
        sim_kwargs['viz_plots']       = viz_plots
        sim_kwargs['worksheet']       = worksheet
        sim_kwargs['row_num']         = test_num

        # Run Montecarlo simulation for chosen parameter test
        run_MC_stochastic_est_SEIR(rollouts, *prob_params, **sim_kwargs)

    workbook.close()

if __name__ == '__main__':
    """ Defined type of probability distributions to sample from:
        gamma, log-Normal (proper distributions)
        gaussian, uniform (not adequate for beta or gamma_{-1})
    """
    run(prob_type = 'gamma', days = 200)
    # run(prob_type = 'log-normal', days = 500)    
    # run(prob_type = 'gaussian', days = 500)
    # run(prob_type = 'uniform', days = 500)