import numpy as np
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import rc

# For custom classes and functions
from epimodels.seir   import *
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


def plot_simulation(S, E, I, R, T,  **kwargs):
    #################################################
    ######## Plotting and storing parameters ########
    #################################################
    x_axis_offset           = 0.375*kwargs['days']
    y_axis_offset           = 0.003
    plot_all = 1; show_S = 1; show_E = 1; show_R = 0; plot_peaks = 1; show_T = 1; 
    scale_offset = 0.025; show_analytic_limit = 0
    
    t = np.arange(0,kwargs['days'])
    SIRvariables   = S, E, I, R, T, t
    Plotoptions    = plot_all, show_S, show_E, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset  
    plotSEIR_evolution(SIRvariables, Plotoptions, **kwargs)

    if kwargs['do_growth']:
        tc_Reff, Rt_tc = plotSEIR_growth((S,t), **kwargs)  

    if kwargs['do_infected']:
        T_limit        = 121   # days for infected plot only
        x_axis_offset  = -60
        Ivariables     = I, t
        Plotoptions    = T_limit, plot_peaks, x_axis_offset, y_axis_offset
        plotInfected_evolution(Ivariables, Plotoptions, **kwargs)    

    # Show all the plots of chosen
    plt.show()


def main():    

    ####################################################################
    ######## Choose Initial and Model Parameters for Simulation ########
    ####################################################################
    '''Simulation options defined in sims.py
        sim_num = 5  --> Primer case study
    '''
    sim_kwargs = loadSimulationParams(5, 0, plot_data = 1, header = 'SEIR')
    sim_kwargs['do_infected'] = 0
    
    #####################################################
    ######## Simulate Vanilla SIR Model Dynamics ########
    #####################################################
    S,E,I,R,T = run_SEIR_sim(**sim_kwargs)

    print('*********   Results    *********')    
    tc, _, _, _, _ = getCriticalPointsAfterPeak(I)
    T_tc  = T[tc]
    print('Total Cases @ Peak = ', T_tc,'by day=', tc)
    total_infected     = I[-1]
    print('Infected @ t(end) = ', total_infected)
    total_cases     = T[-1]
    print('Total Cases @ t(end) = ', total_cases)

    ############################################
    ######## Plots of Single Simulation ########
    ############################################    
    plot_simulation(S, E, I, R, T, **sim_kwargs)

if __name__ == '__main__':
    main()