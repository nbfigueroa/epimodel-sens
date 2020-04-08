import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
import xlsxwriter

# Importing models and plotting functions
from epimodels.seiqr import *
from epimodels.utils import *


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

sim_num = 3
eps = 1e-20

# Equation to estimate final epidemic size (infected)
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

############################################
######## Parameters for Simulation ########
############################################
# Initial values from March 4th for India test-case
scenario = 0
N            = 1375987036 # Number from report
days         = 712
gamma_inv    = 7  
m            = 0.0043
tau_q_inv    = 14

# Initial values from March 4th for India test-case
R0           = 3
D0           = 0         
Q0           = 28              

# Derived Model parameters and 
gamma      = 1.0 / gamma_inv
tau_q      = 1.0 /tau_q_inv

# Control variable:  percentage quarantined
q           = 0.001
# Q0 is 1% of total infectious; i.e. I0 + Q0 (as described in report)
# In the report table 1, they write number of Quarantined as SO rather than Q0
# Q0, is this a typo? 
# Number of Infectuos as described in report    
I0          = ((1-q)/(q)) * Q0  

# The initial number of exposed E(0) is not defined in report, how are they computed?
contact_rate = 10                     # number of contacts an individual has per day
E0           = (contact_rate - 1)*I0  # Estimated exposed based on contact rate and inital infected

######### Plotting and storing parameters ##############
x_axis_offset       = 250
y_axis_offset       = 0.0000000003
store_plots         = 1 
plot_all            = 1
plot_peaks          = 1
show_S              = 0
show_R              = 0
show_E              = 0
show_Q              = 1
show_D              = 1
plot_superimposed   = 1
store_values        = 1
show_analytic_limit = 0
plot_r0_dependence  = 0
do_growth           = 0


###### Specify scenarios and beta values to test ################
scenario = 0

# infection rate 
r0                  = 2.28      
beta                = r0 / gamma_inv
# sigma_inv_AFMC    = 5.1
sigma_inv_AFMC    = 11.5

# Variables for beta samples
error_perc      = 15
sigma_inv_error = error_perc/100
sigma_inv_mean  = sigma_inv_AFMC
sigma_inv_sigma = sigma_inv_AFMC*(sigma_inv_error)
sigma_inv_plus  = sigma_inv_mean + sigma_inv_sigma
sigma_inv_minus = sigma_inv_mean - sigma_inv_sigma

sigma_inv_samples = [sigma_inv_mean, sigma_inv_plus, sigma_inv_minus]

######## Record predictions ########
S_samples       = np.empty([3, days])
E_samples       = np.empty([3, days])
I_samples       = np.empty([3, days])
Q_samples       = np.empty([3, days])
Re_samples      = np.empty([3, days])
D_samples       = np.empty([3, days])
file_extensions  = ["./results/AFMC_Scenario{scenario:d}_sigma{error:d}error".format(scenario=scenario, error=error_perc), 
                    "./results/AFMC_Scenario{scenario:d}_sigma{error:d}error_plus".format(scenario=scenario, error=error_perc),
                    "./results/AFMC_Scenario{scenario:d}_sigma{error:d}error_minus".format(scenario=scenario, error=error_perc),]
header = ['sigma', 'R0', 'I_42', 'T_42', 'I_70', 'T_70', 'I_90', 'T_90', 'I_120', 'T_120', 't_low', 't_c', 'I_peak', 'T_inf']
time_checkpoints = [42, 70, 90, 120] 

if store_values:
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(file_extensions[0]+"_sigma.xlsx")
    worksheet = workbook.add_worksheet()
    for jj in range(len(header)):
        worksheet.write(0, jj,  header[jj] )    


######## Start simulations ########
for ii in range(3):
    #####  beta value to evaluate #####  
    sigma_inv = sigma_inv_samples[ii]
    sigma = 1.0 / sigma_inv
    r0    = beta/gamma

    if store_values:
        # Store beta and r
        worksheet.write(ii+1, 0,  sigma )
        worksheet.write(ii+1, 1,  float(r0) )

    print('*****   Hyper-parameters    *****')
    print('N=',N,'days=',days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv, 'tauq_inv (days) = ',tau_q_inv)

    print('*****   Model-parameters    *****')
    print('beta=',beta,'gamma=', gamma, 'sigma', sigma, 'tau_q', tau_q, 'm', m)

    ########################################
    ######## SEIQR Model Simulation ########
    ########################################

    ''' Compartment structure of armed forces SEIR model with deaths
        N = S + E + I + Q + R + D
    '''
    # Initial conditions vector
    S0 = N - E0 - I0 - Q0 - R0 - D0
    y0 = S0, E0, I0, Q0, R0, D0
    print("S0=",S0, "E0=",E0, "I0=",I0, "Q0=",Q0, "R0=",R0, "D0", D0)

    # Simulation Options
    solver_type = 1 # ivp - LSODA

    # Simulate ODE equations
    SEIQRparams = N, beta, gamma, sigma, m, q, tau_q
    sol_ode_timeseries = simulate_seiqrModel(SEIQRparams, solver_type, y0, N, days, 1)

    # Unpack time-series
    t  = sol_ode_timeseries[0]    
    S  = sol_ode_timeseries[1]    
    E  = sol_ode_timeseries[2]    
    I  = sol_ode_timeseries[3]    
    Q  = sol_ode_timeseries[4]    
    Re = sol_ode_timeseries[5]    
    D  = sol_ode_timeseries[6]  

    R     = Re + D + Q
    T     = I + R 
    Inf   = I + Q
    All_I = I + Q + E 


    # Storing run in matri for post-processing
    S_samples[ii,:]  = S
    E_samples[ii,:]  = E
    I_samples[ii,:]  = I
    Q_samples[ii,:]  = Q
    Re_samples[ii,:] = Re
    D_samples[ii,:]  = D

    print('*********   Results    *********')    
    if store_values:
        # Store Intervals
        for jj in range(len(time_checkpoints)):
            worksheet.write(ii+1, (jj+1)*2,  Inf[time_checkpoints[jj]])
            worksheet.write(ii+1, (jj+1)*2 + 1,  T[time_checkpoints[jj]])
            print('Inf(',time_checkpoints[jj],')=',Inf[time_checkpoints[jj]], 'T(',time_checkpoints[jj],')=',T[time_checkpoints[jj]])


    Ids_less_10  = np.nonzero(Inf < 11)
    I_less_10 =  Ids_less_10[0][0]
    print('I(t_low)=',I_less_10)

    if show_analytic_limit:
        # Estimated Final epidemic size (analytic) not-dependent on simulation
        init_guess   = 0.0001
        r0_test      = r0
        SinfN  = fsolve(epi_size, init_guess)
        One_SinfN = 1 - SinfN
        print('*****   Final Epidemic Size    *****')
        print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

    print('*************   Results    *************')
    peak_All_I_idx =  np.argmax(All_I)
    peak_All_I     = All_I[peak_All_I_idx]
    print('Peak Instant. ALL Infectouos = ', peak_All_I,'by day=', peak_All_I_idx)

    peak_infe_idx =  np.argmax(Inf)
    peak_infe     = Inf[peak_infe_idx]
    print('Peak Instant. Infectouos = ', peak_infe,'by day=', peak_infe_idx)

    peak_Q_idx =  np.argmax(Q)
    peak_Q     = Q[peak_Q_idx]
    print('Peak Instant. Quarantined = ', peak_Q,'by day=', peak_Q_idx)

    peak_E_idx =  np.argmax(E)
    peak_E     = E[peak_E_idx]
    print('Peak Instant. Exposed = ', peak_E,'by day=', peak_E_idx)

    peak_inf_idx =  np.argmax(I)
    peak_inf     = I[peak_inf_idx]
    print('Peak Instant. Infected = ', peak_inf,'by day=', peak_inf_idx)

    peak_total_inf  = T[peak_inf_idx]
    print('Total Cases when Peak = ', peak_total_inf,'by day=', peak_inf_idx)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    if store_values:
        # Store results
        worksheet.write(ii+1, 10,  I_less_10)
        worksheet.write(ii+1, 11,  peak_infe_idx)
        worksheet.write(ii+1, 12,  peak_infe)
        worksheet.write(ii+1, 13,  total_cases)

    #####################################################################
    ######## Plots Simulation with point estimates of parameters ########
    #####################################################################

    txt_title = r"COVID-19 AFMC SEIQR Model Dynamics [Scenario 0] ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, 1/$\tau_q$={tau_q_inv:1.2f}, $q$={q:1.4f})"

    SEIQRparams    = scenario, r0, beta, gamma_inv, sigma_inv, tau_q_inv, q, N
    SEIQRvariables = S, E, I, Q, Re ,D , t
    plot_all_ii    = 1
    Plotoptions    = plot_all, show_S, show_E, show_Q, show_R, show_D, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset
    plotSEIQR_evolution(txt_title, SEIQRparams, SEIQRvariables, Plotoptions, store_plots, file_extensions[ii])


if store_values:
    workbook.close()

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
if plot_superimposed:
    SEIQRparams  = scenario, r0, beta, gamma_inv, sigma_inv_AFMC, tau_q_inv, q, N
    plot_peaks_all = 0
    Plotoptions  = plot_all, show_S, show_R, show_analytic_limit, plot_peaks_all, x_axis_offset, y_axis_offset, sigma_inv_error
    text_error   = r"$\sigma \pm %1.2f \sigma $"%sigma_inv_error
    plotSEIQR_evolutionErrors(txt_title, SEIQRparams, S_samples, E_samples, I_samples, Q_samples, Re_samples, D_samples, Plotoptions, text_error, store_plots, file_extensions[0])
    
###########################################################################
######## TO CHECK: Plots Simulation with reproductive/growth rates ########
###########################################################################
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
    ax1.text(t[-1]-x_axis_offset, 1 + 0.01, txt1.format(per=1), fontsize=20, color='r')
    ax1.text(t[-1]-x_axis_offset,2.5, r"${\cal R}_t \equiv \left( \frac{S (t) }{N (t) } \right) {\cal R}_0$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))


    # Estimations of End of Epidemic
    effRT_diff     = effective_Rt - 1
    ids_less_1     = np.nonzero(effRT_diff < 0)
    if len(ids_less_1)> 1:
        effRT_crossing = ids_less_1[0][0]
        ax1.plot(effRT_crossing, 1,'ro', markersize=12)
        ax1.text(effRT_crossing-10, 1-0.2,str(effRT_crossing), fontsize=10, color="r")


    ax1.set_ylabel('Rt (Effective Reproductive Rate)', fontsize=20)
    ax1.set_xlabel('Time[days]', fontsize=20)
    ax1.set_ylim(0,4)
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    txt_title = r"COVID-19 AFMC SEIQR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, 1/$\tau_q$={tau_q_inv:1.3f}, $q$={q:1.3f})"
    fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, tau_q_inv = tau_q_inv, q=q),fontsize=15)

    # Plot of temporal growth rate
    ax2.plot(t, growth_rates, 'k', lw=2, label='rI (temporal growth rate)')
    ax2.text(t[0] + 0.02, growth_rates[0] - 0.02,r'${r}_I(t)$', fontsize=10)    
    ax2.plot(t, 0*np.ones(len(t)), 'r-')
    txt1 = r"Critical ($r_I$={per:2.2f})"
    ax2.text(t[-1]-x_axis_offset, 0 + 0.01, txt1.format(per=0), fontsize=20, color='r')
    ax2.text(t[-1]-x_axis_offset, 0.2, r"$r_I  \equiv \gamma \left[ {\cal R}_t - 1 \right]$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
    ax2.text(t[-1]-x_axis_offset, 0.1, r"$\frac{ dI}{dt} = r_I \, I $", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
    
    ax2.set_ylabel('rI (temporal growth rate)', fontsize=20)
    ax2.set_xlabel('Time[days]',fontsize=20)
    ax2.set_ylim(-0.2,0.5)


    # Estimations of End of Epidemic
    rI_diff     = growth_rates 
    ids_less_0  = np.nonzero(rI_diff < 0)
    print(ids_less_0)
    if len(ids_less_0) > 1:
        rI_crossing = ids_less_1[0][0]
        ax2.plot(rI_crossing, 0,'ro', markersize=12)
        ax2.text(rI_crossing-10, 0-0.04,str(rI_crossing), fontsize=10, color="r")


    fig.set_size_inches(27.5/2, 12.5/2, forward=True)

    plt.savefig('./figures/AFMCSIR_growthRates_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./figures/AFMCSIR_growthRates_%i.pdf'%sim_num, bbox_inches='tight')


plt.show()


