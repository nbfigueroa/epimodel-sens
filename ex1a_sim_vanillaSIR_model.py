import numpy as np
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import rc

# For custom classes and functions
from epimodels.sir    import *
from epimodels.utils  import *
from epimodels.plots  import *
from epimodels.sims   import *


def run_SIR_sim(**kwargs):
    '''
        Run a single simulation of SIR dynamics
    '''
    N          = kwargs['N']
    days       = kwargs['days']
    beta       = kwargs['beta']
    r0         = kwargs['r0']
    gamma_inv  = kwargs['gamma_inv']
    gamma      = 1.0 / gamma_inv

    print('*****   SIMULATING SIR MODEL DYNAMICS *****')    
    print('*****         Hyper-parameters        *****')
    print('N=',N,'days=', days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)
    print('*****         Model-parameters        *****')
    print('beta=',beta,'gamma=',gamma)

    model_kwargs = {}
    model_kwargs['r0']         = r0
    model_kwargs['inf_period'] = gamma_inv
    model_kwargs['I0']         = kwargs['I0']
    model_kwargs['R0']         = kwargs['R0']

    model   = SIR(N,**model_kwargs)
    S,I,R   = model.project(days,'ode_int')
    T       = I + R    
    return S,I,R,T


def plot_simulation(S, I, R, T, t, **kwargs):
    #################################################
    ######## Plotting and storing parameters ########
    #################################################
    x_axis_offset           = 0.375*kwargs['days']
    y_axis_offset           = 0.003
    plot_all = 1; show_S = 1; show_R = 0; plot_peaks = 1; show_T = 1; scale_offset = 0.025; show_analytic_limit = 0
    
    SIRvariables   = S, I, R, T, t
    Plotoptions    = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset  
    plotSIR_evolution(SIRvariables, Plotoptions, **kwargs)

    if kwargs['do_growth']:
        tc_Reff, Rt_tc, tc_growth, rI_tc = plotSIR_growth((S, t), **kwargs)  

    T_limit        = 121   # days for infected plot only
    x_axis_offset  = -60
    Ivariables     = I, t
    Plotoptions    = T_limit, plot_peaks, x_axis_offset, y_axis_offset
    plotInfected_evolution(Ivariables, Plotoptions, **kwargs)
    show_analytic_limit  = 1
    if show_analytic_limit:
        plotSIR_finalEpidemicR0(**kwargs)

    # Show all the plots of chosen
    plt.show()


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
    sim_num    = 5;
    sim_kwargs = loadSimulationParams(sim_num, 0)

    #####################################################
    ######## Simulate Vanilla SIR Model Dynamics ########
    #####################################################
    S,I,R,T = run_SIR_sim(**sim_kwargs)

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
    t = np.arange(0,sim_kwargs['days'])
    plot_simulation(S, I, R, T, t, **sim_kwargs)

if __name__ == '__main__':
    main()