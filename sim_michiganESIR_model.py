import numpy as np
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import rc

# For custom classes and functions
from epimodels.esir import *
from epimodels.utils import *


# To read data from xls sheet
from pandas import DataFrame, read_csv
import pandas as pd 

# For beautiful plots
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Select simulation number
sim_num = 2

#################################################################
######## Set Initial and Model Parameters for Simulation ########
#################################################################
# India Simulation with Cambridge/Michigan Data Starting March 4th
if sim_num == 1:
    scenarios       = [0,1,2]
    time_points     = [[],[11,15,21,42],[11,15,21,70]]
    pi_values       = [[],[0.8,0.6,0.2,1],[0.8,0.6,0.2,1]]
    check_points    = np.array([11,15,21,42])
    scenario        = 2
    N               = 1353344709
    days            = round(30.5*10)
    gamma_inv       = 7  
    r0              = 2.28  
    I0, R0          = 25, 3    
    beta            = r0 / gamma_inv
    file_extension  = "./results/MichiganESIR_Scenario{scenario:d}_India_{days:d}".format(scenario=scenario, days = days)
    x_tick_names    = ('4 Mar', '4 Apr', '4 May', '4 Jun', '4 Jul', '4 Aug', '4 Sept', '4 Oct', '4 Nov', '4 Dec', '4 Jan')
    figure_title    = r"COVID-19 Michigan eSIR Model Dynamics [Scenario {scenario:d}] --India-- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
    Infected_conf   = np.array([])
    Infected_est    = np.array([])

# Yucatan Simulation Starting March 13th
if sim_num == 2:

    ''' Description of Scenario 1 [Hypothetical]:
        - On March 16th (day 4) schools closed; i.e. 40-50% (https://www.populationpyramid.net/mexico/2019/)
        - On March 23rd (day 11) lock-down imposed until April 30 [38 day lock-down]
        Description of Scenario 2 [More-realistic]:
        - On March 23rd (day 11) schools really closed
    ''' 

    scenarios       = [0, 1, 2]
    time_points     = [[],[4, 11, 18, 48],     [4, 18, 48]]
    pi_values       = [[],[0.8, 0.6, 0.2,  1], [0.8, 0.4, 1]]
    check_points    = np.array([4, 11, 18, 48])
    # check_points    = np.array([])

    scenario        = 2
    N               = 2000000 
    days            = round(30.5*7)
    gamma_inv       = 7
    # R0 estimates 2.28 (2.06–2.52) https://doi.org/10.1016/j.ijid.2020.02.033                
    r0              = 2.52   
    I0, R0          = 1, 0
    file_extension  = "./results/yucatan/MichiganESIR_Scenario{scenario:d}_Yucatan_{days:d}".format(scenario=scenario, days = days)
    x_tick_names    = ('13 Mar', '13 Apr', '13 May', '13 Jun', '13 Jul', '13 Aug', '13 Sept', '13 Oct')
    figure_title   = r"COVID-19 Michigan eSIR Model Dynamics [Scenario {scenario:d}] --Yucatan, Mexico-- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"

    # Load number of infected cases
    file           = r'./data/covid_yucatan.xlsx'
    case_data      = pd.read_excel(file)
    casos          = case_data['Unnamed: 1']
    muertes        = case_data['Unnamed: 5']
    centinal       = case_data['Unnamed: 3']
    Infected_conf  = np.array(casos[1:])
    Dead           = np.array(muertes[1:])
    Infected_est   = np.array(centinal[1:])

# Derived values
beta       = r0 / gamma_inv
gamma      = 1.0 / gamma_inv

#################################################
######## Plotting and storing parameters ########
#################################################
x_axis_offset           = 0.375*days
y_axis_offset           = 0.003
store_plots             = 1
plot_all                = 0
show_S                  = 0
show_R                  = 0
plot_peaks              = 1
show_T                  = 1
show_analytic_limit     = 0
do_growth               = 0
scale_offset            = 0.025 

#######################################################
######## Simulation Vanilla SIR Model Dynamics ########
#######################################################

print('*****   SIMULATING ESIR MODEL DYNAMICS *****')    
print('*****         Hyper-parameters        *****')
print('N=',N,'days=',days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)
print('*****         Model-parameters        *****')
print('beta=',beta,'gamma=',gamma)

kwargs = {}
kwargs['r0'] = r0
kwargs['inf_period'] = gamma_inv
kwargs['I0'] = I0
kwargs['R0'] = R0

model   = eSIR(N, time_points[scenario], pi_values[scenario], **kwargs)
S,I,R,t = model.project(days)
T       = I + R

pi_t = np.zeros([t.size])
for tt in range(t.size):
    pi_t[tt] = model.determine_scaling(tt)

print(pi_t)
print('*********   Results    *********')    
tc, t_I100, t_I500, t_I100, t_I10 = getCriticalPointsAfterPeak(I)
T_tc  = T[tc]
print('Total Cases @ Peak = ', T_tc,'by day=', tc)
total_infected     = I[-1]
print('Infected @ t(end) = ', total_infected)
total_cases     = T[-1]
print('Total Cases @ t(end) = ', total_cases)

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
SIRparams      = scenario, float(r0), beta, gamma, N
SIRvariables   = S, I, R, T, t
Plotoptions    = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, store_plots
number_scaling = 'million'
x_tick_labels  = np.arange(0, days, 30), x_tick_names
plotSIR_evolution(figure_title, SIRparams, SIRvariables, Plotoptions, file_extension, number_scaling,  x_tick_labels)

if scenario == 0:
    T_limit    = days   # days for infected plot only
else: 
    T_limit    = 61   # days for infected plot only

number_scaling = 'none'

x_axis_offset  = 0
Ivariables     = I, t
plot_peaks     = 0
Plotoptions    = T_limit, plot_peaks, x_axis_offset, y_axis_offset, store_plots
x_tick_labels  = np.arange(0, T_limit, 30), x_tick_names
plotInfected_evolution(figure_title, SIRparams, Ivariables, Plotoptions, file_extension, number_scaling,  x_tick_labels, Infected_conf, Infected_est, check_points, pi_t)


plt.show()




