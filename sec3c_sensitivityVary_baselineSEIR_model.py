import numpy as np
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import rc

# For custom classes and functions
from epimodels.seir    import *
from epimodels.utils  import *
from epimodels.plots  import *
from epimodels.sims   import *


def run_SEIR_sim(**kwargs):
    '''
        Run a single simulation of SEIR dynamics
    '''
    N          = kwargs['N']
    days       = kwargs['days']
    beta       = kwargs['beta']
    r0         = kwargs['r0']
    gamma_inv  = kwargs['gamma_inv']    
    gamma      = 1.0 / gamma_inv
    sigma_inv  = kwargs['sigma_inv']
    sigma      = 1.0 / sigma_inv

    print('*****   SIMULATING SEIR MODEL DYNAMICS *****')    
    print('*****         Hyper-parameters        *****')
    print('N=',N, 'days=', days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv, 'sigma_inv (days) = ', sigma_inv)
    print('*****         Model-parameters        *****')
    print('beta=', beta, 'gamma=', gamma, 'sigma=', sigma)


    # Populate parameters dictionary
    model_kwargs = {}    

    # Initial parameters
    model_kwargs['I0']         = kwargs['I0']
    model_kwargs['E0']         = kwargs['E0']
    model_kwargs['R0']         = kwargs['R0']

    # Model parameters
    model_kwargs['r0']         = r0    
    model_kwargs['beta']       = beta
    model_kwargs['gamma']      = gamma
    model_kwargs['sigma']      = sigma

    model   = SEIR(N,**model_kwargs)
    S,E,I,R = model.project(days,'ode_int')
    T       = I + R    
    return S,E,I,R,T



def storeVariationResults(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, I_samples, R_samples, **kwargs):
    
    T_samples  = I_samples+ R_samples
    R0_samples = np.array(beta_samples) * np.array(gamma_inv_samples)
    tc         = np.argmax(I_samples, axis=1)
    
    # Store stats
    worksheet        = kwargs['worksheet']
    row_num          = kwargs['row_num']

    tests, days = T_samples.shape
    tc_samples = []; Ipeak_samples = []; Tend_samples = [];
    for test in range(tests):      
        tc_samples.append(tc[test])
        Ipeak_samples.append(I_samples[test,tc[test]])
        Tend_samples.append(T_samples[test,days-1])

    worksheet.write_row(row_num, 0,  beta_samples)
    worksheet.write_row(row_num, 3,  gamma_inv_samples)
    worksheet.write_row(row_num, 6,  sigma_inv_samples)
    worksheet.write_row(row_num, 9,  E0_samples)
    worksheet.write_row(row_num, 12,  R0_samples)    
    worksheet.write_row(row_num, 15, tc_samples)
    worksheet.write_row(row_num, 18, Ipeak_samples)
    worksheet.write_row(row_num, 21, Tend_samples)



def run_SEIR_wErrors(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, text_error, **kwargs):
    '''
        Run multiple SEIR simulations (means +/- errors)
    '''
    scenario = kwargs['scenario']

    ######## Record predictions ########
    S_samples       = np.empty([3, kwargs['days']])
    E_samples       = np.empty([3, kwargs['days']])
    I_samples       = np.empty([3, kwargs['days']])
    R_samples       = np.empty([3, kwargs['days']])

    ############################################################
    ######## Simulate Single Vanilla SIR Model Dynamics ########
    ############################################################
    for ii in range(len(beta_samples)):
        kwargs['beta']      = beta_samples[ii]
        kwargs['gamma_inv'] = gamma_inv_samples[ii]
        kwargs['sigma_inv'] = sigma_inv_samples[ii]
        kwargs['r0']        = beta_samples[ii]*gamma_inv_samples[ii]
        kwargs['E0']        = E0_samples[ii]

        S,E,I,R,T             = run_SEIR_sim(**kwargs)

        # Storing run in matrix for post-processing
        S_samples[ii,:] = S
        E_samples[ii,:] = E
        I_samples[ii,:] = I
        R_samples[ii,:] = R


    storeVariationResults(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, I_samples, R_samples, **kwargs)

    ##############################################################
    ######## Plots Simulation Variables with Error Bounds ########
    ##############################################################
    x_axis_offset       = round(kwargs['days']*0.4)
    y_axis_offset       = 0.0000000003 
    plot_all            = 1; plot_peaks = 1; show_S = 0; show_T = 1; show_R = 0; show_analytic_limit = 0; scale_offset = 0.01 
    Plotoptions         = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset, scenario
    plotSIR_evolutionErrors_new(S_samples, I_samples, R_samples, Plotoptions, text_error, **kwargs)    


def main():    

    ####################################################################
    ######## Choose Initial and Model Parameters for Simulation ########
    ####################################################################
    '''Simulation options defined in sims.py
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Primer case study
    '''
    sim_num    = 5; scenario   = 0
    sim_kwargs                 = loadSimulationParams(sim_num, scenario, plot_data = 0, header = 'SEIR')
    # Need to get rid of this variable here/..
    sim_kwargs['scenario']     = scenario
    basefilename               = sim_kwargs['file_extension']
    workbook, worksheet        = createResultsfile(basefilename, 'errorVary', test_type='varying')

    ## For variation on these parameters
    beta       = sim_kwargs['beta']
    gamma_inv  = sim_kwargs['gamma_inv']
    sigma_inv  = sim_kwargs['sigma_inv']
    E0         = sim_kwargs['E0']

    # Variables for +/- errors on beta
    error_perc        = 10
    err               = error_perc/100
    
    ########### Test 1: Vary beta, fix gamma and sigma ############
    text_error                    = r"$\beta \pm %1.2f \beta $"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVaryBeta"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 1
    beta_samples                  = [beta, beta*(1+err), beta*(1-err)]
    gamma_inv_samples             = [gamma_inv, gamma_inv, gamma_inv]
    sigma_inv_samples             = [sigma_inv, sigma_inv, sigma_inv]
    E0_samples                    = [E0, E0, E0]
    run_SEIR_wErrors(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, text_error, **sim_kwargs)
    plt.show()

    ########### Test 2: fix beta, sigma, vary gamma ############
    text_error                    = r"$\gamma^{-1} \pm %1.2f \gamma^{-1} $"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVaryGamma"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 2
    beta_samples                  = [beta, beta, beta]
    gamma_inv_samples             = [gamma_inv, gamma_inv*(1+err), gamma_inv*(1-err)]
    sigma_inv_samples             = [sigma_inv, sigma_inv, sigma_inv]
    E0_samples                    = [E0, E0, E0]
    run_SEIR_wErrors(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, text_error, **sim_kwargs)
    plt.show()

    ########### Test 3: vary sigma, fix beta and gamma ############
    text_error                    =  r"$\sigma^{-1} \pm %1.2f \sigma^{-1}$"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVarySigma"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 3
    beta_samples                  = [beta, beta, beta]
    gamma_inv_samples             = [gamma_inv, gamma_inv, gamma_inv]
    sigma_inv_samples             = [sigma_inv, sigma_inv*(1+err), sigma_inv*(1-err)]
    E0_samples                    = [E0, E0, E0]
    run_SEIR_wErrors(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, text_error, **sim_kwargs)
    plt.show()
    
    ########### Test 3: vary sigma, fix beta and gamma ############
    text_error                    =  r"$E(0) \pm %1.2f E(0)$"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVarySigma"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 3
    beta_samples      = [beta, beta, beta]
    gamma_inv_samples = [gamma_inv, gamma_inv, gamma_inv]
    sigma_inv_samples = [sigma_inv, sigma_inv, sigma_inv]
    E0_samples        = [E0, E0*(1+err), E0*(1-err)]
    run_SEIR_wErrors(beta_samples, gamma_inv_samples, sigma_inv_samples, E0_samples, text_error, **sim_kwargs)
    plt.show()

    workbook.close()

if __name__ == '__main__':
    main()
