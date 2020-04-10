import numpy as np
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import rc

# For custom classes and functions
from epimodels.sir import *
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
    scenario     = 0
    N               = 1353344709
    days            = round(30.5*10)
    gamma_inv       = 7  
    r0              = 2.28  
    I0, R0          = 25, 3    
    beta            = r0 / gamma_inv
    file_extension  = "./results/VanillaSIR_Scenario{scenario:d}_India_{days:d}".format(scenario=scenario, days = days)
    x_tick_names    = ('4 Mar', '4 Apr', '4 May', '4 Jun', '4 Jul', '4 Aug', '4 Sept', '4 Oct', '4 Nov', '4 Dec', '4 Jan')
    figure_title    = r"COVID-19 SIR Model Dynamics [Scenario {scenario:d}] --India-- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
    Infected_conf   = np.array([])
    Infected_est    = np.array([])

# Yucatan Simulation Starting March 13th
# R0 estimates 2.28 (2.06–2.52) https://doi.org/10.1016/j.ijid.2020.02.033
if sim_num == 2:
    scenario        = 0
    N               = 2000000 
    days            = round(30.5*7)
    gamma_inv       = 7                
    r0              = 2.28    
    I0, R0          = 1, 0
    file_extension  = "./results/yucatan/VanillaSIR_Scenario{scenario:d}_Yucatan_{days:d}".format(scenario=scenario, days = days)
    x_tick_names    = ('13 Mar', '13 Apr', '13 May', '13 Jun', '13 Jul', '13 Aug', '13 Sept', '13 Oct')
    figure_title   = r"COVID-19 SIR Model Dynamics [Scenario {scenario:d}] --Yucatan, Mexico-- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"

    # Load number of infected cases
    file        = r'./data/covid_yucatan.xlsx'
    case_data   = pd.read_excel(file)
    casos       = case_data['Unnamed: 1']
    muertes     = case_data['Unnamed: 5']
    centinal    = case_data['Unnamed: 3']
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

print('*****   SIMULATING SIR MODEL DYNAMICS *****')    
print('*****         Hyper-parameters        *****')
print('N=',N,'days=',days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)
print('*****         Model-parameters        *****')
print('beta=',beta,'gamma=',gamma)

kwargs = {}
kwargs['r0'] = r0
kwargs['inf_period'] = gamma_inv
kwargs['I0'] = I0
kwargs['R0'] = R0

model   = SIR(N,**kwargs)
S,I,R,t = model.project(days,'ode_int')
T       = I + R

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

T_limit        = 121   # days for infected plot only
x_axis_offset  = -60
Ivariables     = I, t
Plotoptions    = T_limit, plot_peaks, x_axis_offset, y_axis_offset, store_plots
x_tick_labels  = np.arange(0, T_limit, 30), x_tick_names
plotInfected_evolution(figure_title, SIRparams, Ivariables, Plotoptions, file_extension, number_scaling,  x_tick_labels, Infected_conf, Infected_est)


#################################################
######## Plots Reproductive/growth rates ########
#################################################
if do_growth:
    # Analytic growth rate
    effective_Rt = r0 * (S/N)
    growth_rates = gamma * (effective_Rt - 1)

    ####### Plots for Growth Rates #######
    fig, (ax1, ax2) = plt.subplots(1,2)

    # Plot of Reproductive rate (number)
    ax1.plot(t, effective_Rt, 'k', lw=2, label='Rt (Effective Reproductive Rate)')
    ax1.text(t[0] + 0.02, effective_Rt[0] - 0.15,r'${\cal R}_t$', fontsize=10)
    ax1.plot(t, 1*np.ones(len(t)), 'r-')
    txt1 = "Critical (Rt={per:2.2f})"
    ax1.text(t[-1]-50, 1 + 0.01, txt1.format(per=1), fontsize=12, color='r')
    ax1.text(t[-1]-50,2.5, r"${\cal R}_t \equiv \left( \frac{S (t) }{N (t) } \right) {\cal R}_0$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))


    # Estimations of End of Epidemic
    effRT_diff     = effective_Rt - 1
    ids_less_1     = np.nonzero(effRT_diff < 0)
    effRT_crossing = ids_less_1[0][0]
    ax1.plot(effRT_crossing, 1,'ro', markersize=12)
    ax1.text(effRT_crossing-10, 1-0.2,str(effRT_crossing), fontsize=10, color="r")
    ax1.set_ylabel('Rt (Effective Reproductive Rate)', fontsize=12)
    ax1.set_xlabel('Time[days]', fontsize=12)
    ax1.set_ylim(0,4)
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    txt_title = "COVID-19 SIR Model Dynamics (N={N:2.0f},R0={R0:1.3f},1/gamma={gamma_inv:1.3f}, beta={beta:1.3f})"
    fig.suptitle(txt_title.format(N=N, R0=r0, gamma_inv = gamma_inv, beta= beta),fontsize=15)

    # Plot of temporal growth rate
    ax2.plot(t, growth_rates, 'k', lw=2, label='rI (temporal growth rate)')
    ax2.text(t[0] + 0.02, growth_rates[0] - 0.02,r'${r}_I(t)$', fontsize=10)

    ax2.plot(t, 0*np.ones(len(t)), 'r-')
    txt1 = r"Critical ($r_I$={per:2.2f})"
    ax2.text(t[-1]-50, 0 + 0.01, txt1.format(per=0), fontsize=12, color='r')
    ax2.text(t[-1]-50, 0.2, r"$r_I  \equiv \gamma \left[ {\cal R}_t - 1 \right]$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
    ax2.text(t[-1]-50, 0.1, r"$\frac{ dI}{dt} = r_I \, I $", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
    ax2.set_ylabel('rI (temporal growth rate)', fontsize=12)
    ax2.set_xlabel('Time[days]',fontsize=12)
    ax2.set_ylim(-0.2,0.5)


    # Estimations of End of Epidemic
    rI_diff     = growth_rates 
    ids_less_0  = np.nonzero(rI_diff < 0)
    rI_crossing = ids_less_1[0][0]
    ax2.plot(rI_crossing, 0,'ro', markersize=12)
    ax2.text(rI_crossing-10, 0-0.04,str(rI_crossing), fontsize=10, color="r")
    fig.set_size_inches(27.5/2, 12.5/2, forward=True)

    plt.savefig('./snaps/vanillaSIR_growthRates_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./snaps/vanillaSIR_growthRates_%i.pdf'%sim_num, bbox_inches='tight')



if show_analytic_limit:
    #############################################################
    ######## Dependence of R0 on Final Epidemic Behavior ########
    #############################################################

    # Equation to estimate final epidemic size (infected)
    def epi_size(x):
        return np.log(x) + r0_test*(1-x)

    # Final epidemic size (analytic)
    r0_vals     = np.linspace(1,5,100) 
    init_guess  = 0.0001
    Sinf_N      =   []
    Sinf_S0     =   []
    for ii in range(len(r0_vals)):
        r0_test = r0_vals[ii]
        Sinf_N.append(fsolve(epi_size, init_guess))     
        Sinf_S0.append(1 - Sinf_N[ii])

    r0_test      = r0
    covid_SinfN  = fsolve(epi_size, init_guess)
    covid_SinfS0 = 1 - covid_SinfN
    print('Covid r0 = ', r0_test, 'Covid Sinf/S0 = ', covid_SinfN[0], 'Covid Sinf/S0 = ', covid_SinfS0[0]) 

    # Plots
    fig0, ax0 = plt.subplots()
    ax0.plot(r0_vals, Sinf_S0, 'r', lw=2, label='Susceptible')
    ax0.set_ylabel('$S_{\infty}/S_{0}$ (percentage of population infected)', fontsize=12)
    ax0.set_xlabel('$R_0$', fontsize=12)

    # Current estimate of Covid R0
    plt.title('Final Size of Epidemic Dependence on $R_0$ estimate',fontsize=15)
    ax0.plot(r0_test, covid_SinfS0, 'ko', markersize=5, lw=2)

    # Plot mean
    txt = 'Covid R0({r0:3.3f})'
    ax0.text(r0_test - 0.45, covid_SinfS0 + 0.05,txt.format(r0=r0_test), fontsize=10)
    plt.plot([r0]*10,np.linspace(0,covid_SinfS0,10), color='black')
    txt = "{Sinf:3.3f} Infected"
    ax0.text(1.1, covid_SinfS0 - 0.025,txt.format(Sinf=covid_SinfS0[0]), fontsize=8)
    plt.plot(np.linspace(1,[r0],10), [covid_SinfS0]*10, color='black')

    ax0.text(4, 0.75, r"${\cal R}_0 \equiv \frac{ \beta } {\gamma}$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
    fig0.set_size_inches(18.5/2, 12.5/2, forward=True)
    plt.savefig('./snaps/vanillaSIR_finalSize_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./snaps/vanillaSIR_finalSize_%i.pdf'%sim_num, bbox_inches='tight')
    # plt.close()


plt.show()




