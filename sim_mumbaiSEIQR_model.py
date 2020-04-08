import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc


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

# Simulation for a toy problem for implementation comparison and sanity checks
if sim_num == 1:
    
    # Simulation parameters
    S0   = 9990
    I0   = 1
    R0   = 0
    E0   = 9
    D0   = 0
    N    = S0 + I0 + R0 + E0 + D0
    days = 100
    
    # Usual SEIR Model parameters
    gamma_inv = 5                   # infectious period
    sigma_inv = 2                   # latent period 
    m         = 0.0043              # mortality rate

    # Typical equation for beta computation if estimates are known
    contact_rate = 10               # number of contacts per day
    transmission_probability = 0.07 # transmission probability
    beta         = contact_rate * transmission_probability

    # Derived Model parameters and Control variable     
    gamma  = 1 / gamma_inv
    sigma  = 1 / sigma_inv
    r0     = beta / gamma

    # Extra variables for quarentine compartment
    Q0         = 0                 # number of quarantined indivuals
    tau_q_inv  = 7                # quarantined period
    tau_q      = 1.0 /tau_q_inv    # translated to rate
    # tau_q      = tau_q_inv    # translated to rate

    # Control variable:  percentage quarantined
    q      = 0.10
    # Q0 is 1% of total infectious; i.e. I0 + Q0 (as described in report)
    # In the report table 1, they write number of Quarantined as SO rather than Q0
    # Q0, is this a typo? 
    # Number of Infectuos as described in report    
    I0          = ((1-q)/q) * Q0  

    # Plotting
    x_axis_offset = 50

# Initial values from March 21st for India test-case
if sim_num == 2:
    scenario = 0
    # Values used for the indian armed forces model
    # Initial values from March 21st for India test-case
    N            = 1375987036 # Number from report
    days         = 712
    gamma_inv    = 7  
    sigma_inv    = 5.1
    m            = 0.0043
    r0           = 2.28      
    tau_q_inv    = 14

    # Initial values from March 21st "indian armed forces predictions"
    R0           = 23
    D0           = 5         
    Q0           = 249               
    T0           = 334               # This is the total number of confirmed cases for March 21st, not used it seems?                                   

    # Derived Model parameters and 
    beta       = r0 / gamma_inv
    sigma      = 1.0 / sigma_inv
    gamma      = 1.0 / gamma_inv
    tau_q      = 1.0 /tau_q_inv
    # tau_q      = tau_q_inv

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

    # Plotting
    x_axis_offset = 100


# Initial values from March 4th for India test-case
if sim_num == 3:
    scenario = 0
    # Values used for the indian armed forces model
    # Initial values from March 21st for India test-case
    N            = 1375987036 # Number from report
    days         = 712
    gamma_inv    = 7  
    sigma_inv    = 5.1
    m            = 0.0043
    r0           = 2.28      
    tau_q_inv    = 14

    # Initial values from March 4th for India test-case
    R0           = 3
    D0           = 0         
    Q0           = 28              

    # Derived Model parameters and 
    beta       = r0 / gamma_inv
    sigma      = 1.0 / sigma_inv
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

    # Plotting
    x_axis_offset = 100


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

R   = Re + D + Q
T     = I + R 
Inf   = I + Q
All_I = I + Q + E 

# Estimated Final epidemic size (analytic) not-dependent on simulation
init_guess   = 0.0001
r0_test      = r0
SinfN  = fsolve(epi_size, init_guess)
One_SinfN = 1 - SinfN
print('*****   Final Epidemic Size    *****')
print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

print('*****   Results    *****')
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

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################

txt_title = r"COVID-19 Mumbai SEIQR Model Dynamics [Scenario 0] ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, 1/$\tau_q$={tau_q_inv:1.2f}, $q$={q:1.4f})"
filename = './figures/mumbaiSEIQR_timeEvolution_%i'%sim_num

SEIQRparams    = scenario, r0, beta, gamma_inv, sigma_inv, tau_q_inv, q, N
SEIQRvariables = S, E, I, Q, Re ,D , t
Plotoptions    = plot_all, show_S, show_E, show_Q, show_R, show_D, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset
plotSEIQR_evolution(txt_title, SEIQRparams, SEIQRvariables, Plotoptions, store_plots, filename)

#################################################################
######## Plots Simulation with reproductive/growth rates ########
#################################################################
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
    txt_title = r"COVID-19 Mumbai SEIQR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, 1/$\tau_q$={tau_q_inv:1.3f}, $q$={q:1.3f})"
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

    plt.savefig('./figures/mumbaiSIR_growthRates_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./figures/mumbaiSIR_growthRates_%i.pdf'%sim_num, bbox_inches='tight')


#############################################################
######## Dependence of R0 on Final Epidemic Behavior ########
#############################################################

filename = './figures/mumbaiSEIQR_finalSize_%i'%sim_num
if plot_r0_dependence:
    # Final epidemic size (analytic)
    r0_vals     = np.linspace(1,5,100) 
    init_guess  = 0.0001
    Sinf_N      =   []
    Sinf_S0     =   []
    for ii in range(len(r0_vals)):
        r0_test = r0_vals[ii]
        Sinf_N.append(fsolve(epi_size, init_guess))     
        Sinf_S0.append(1 - Sinf_N[ii])

    # Plots
    fig0, ax0 = plt.subplots()
    ax0.plot(r0_vals, Sinf_S0, 'r', lw=2, label='Susceptible')
    ax0.set_ylabel('$1 - S_{\infty}/N$ (Fraction of Population infected)', fontsize=20)
    ax0.set_xlabel('$R_0$', fontsize=20)

    # Current estimate of Covid R0
    plt.title('Final Size of Epidemic Dependence on $R_0$ estimate',fontsize=15)
    ax0.plot(r0, One_SinfN, 'ko', markersize=5, lw=2)

    # Plot mean
    txt = 'Covid-19 R0({r0:3.3f})'
    ax0.text(r0 - 0.45, One_SinfN + 0.05,txt.format(r0=r0, fontsize=10))
    plt.plot([r0]*10,np.linspace(0,One_SinfN,10), color='black')
    txt = "{Sinf:3.3f} Infected"
    ax0.text(1.1, One_SinfN - 0.025,txt.format(Sinf=One_SinfN[0]), fontsize=8)
    plt.plot(np.linspace(1,[r0],10), [One_SinfN]*10, color='black')

    ax0.text(4, 0.75, r"${\cal R}_0 \equiv \frac{ \beta } {\gamma}$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
    fig0.set_size_inches(18.5/2, 12.5/2, forward=True)
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')


plt.show()


