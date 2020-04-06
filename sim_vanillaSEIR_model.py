import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Select simulation number
sim_num = 0

############################################
######## Parameters for Simulation ########
############################################

# Simulation for a toy problem for implementation comparison
if sim_num < 2:
    # Simulation parameters
    S0   = 9990
    I0   = 1
    R0   = 0
    E0   = 9    
    N    = S0 + I0 + R0 + E0 
    days = 100
    
    # Model parameters
    contact_rate = 10                 # number of contacts per day
    transmission_probability = 0.07   # transmission probability
    gamma_inv = 5                     # infectious period
    sigma_inv = 2                     # latent period 
    
    # Derived Model parameters and Control variable 
    beta   = contact_rate * transmission_probability
    gamma  = 1 / gamma_inv
    sigma  = 1 / sigma_inv
    r0     = beta / gamma

    if sim_num == 1:
        D0   = 0
        m    = 0.0043                 # mortality rate
        N    = S0 + I0 + R0 + E0 + D0


# Values used for the indian armed forces model
# Initial values from March 21st for India test-case
if sim_num == 2:
    # Values used for the indian armed forces model
    # Initial values from March 21st for India test-case
    N            = 1375987036 # Number from report
    days         = 356
    gamma_inv    = 7  
    sigma_inv    = 5.1
    m            = 0.0043
    r0           = 2.28      
    tau_q_inv    = 14

    # Initial values from March 21st "indian armed forces predictions"
    R0           = 23
    D0           = 5         
    Q0           = 249             # Q0 is 1% of total infectious; i.e. I0 + Q0 (as described in report)
                                   # In the report table 1, they write number of Quarantined as SO rather than Q0
                                   # Q0, is this a typo? 
    T0           = 334             # This is the total number of comfirmed cases for March 21st, not used it seems?                                   
    I0           = (1.01/0.01) * Q0  # Number of Infectuos as described in report

    # The initial number of exposed E(0) is not defined in report, how are they computed?
    contact_rate = 10                     # number of contacts an individual has per day
    E0           = (contact_rate - 1)*I0  # Estimated exposed based on contact rate and inital infected

    # Derived Model parameters and 
    beta       = r0 / gamma_inv
    sigma      = 1.0 / sigma_inv
    gamma      = 1.0 / gamma_inv
    tau_q      = 1.0 /tau_q_inv

    # Control variable:  percentage quarantined
    q           = 0.01



print('*****   Hyper-parameters    *****')
print('N=',N,'days=',days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)

print('*****   Model-parameters    *****')
print('beta=',beta,'gamma=', gamma, 'sigma', sigma)

######################################
######## Simulation Functions ########
######################################
# Equation to estimate final epidemic size (infected)
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

# The SEIR model differential equations.
def seir_ivp(t, X, N, beta, gamma, sigma):
    S, E, I, R = X
    
    # Main state variables with exponential rates
    dSdt = -(beta * I * S)/N 
    dEdt =  (beta*S*I)/N - sigma*E
    dIdt  = sigma*E - gamma*I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt

def seir_ode(X, t, N, beta, gamma, sigma):
    S, E, I, R = X
    
    # Main state variables with exponential rates
    dSdt = -(beta * I * S)/N 
    dEdt =  (beta*S*I)/N - sigma*E
    dIdt  = sigma*E - gamma*I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt

# The SEIR model differential equations with mortality rates
def seir_deaths_ivp(t, X, N, beta, gamma, sigma, m):
    S, E, I, R, D = X

    dSdt  = - (beta*S*I)/N 
    dEdt  = (beta*S*I)/N - sigma*E    
    
    # Original formulation
    # dIdt  = sigma*E - gamma*I
    # dRdt  = (1-m)*gamma*I 
    # dDdt  = m*gamma*I 

    # Change of variable way --> Yields same results
    dIdt  = sigma*E - gamma*I - m*I
    dRdt  = gamma*I 
    dDdt  = m*I 

    return dSdt, dEdt, dIdt, dRdt, dDdt

def seir_deaths_ode(X, t, N, beta, gamma, sigma, m):
    return seir_deaths_ivp(t, X, N, beta, gamma, sigma, m)

def simulate_seirModel(seir_type, solver_type, y0, simulation_time, dt):
    
    # A grid of time points (in days)
    t_eval = np.arange(0, simulation_time, dt)

    if seir_type == 0:
        # Standard SEIR model no deaths
        ''' Compartment structure of armed forces SEIR model
            N = S + E + I + R 
        '''
        if solver_type:
            # Integrate the SEIR equations with LSODA approach (adaptive)
            ode_sol = solve_ivp(lambda t, X: seir_ivp(t, X, N, beta, gamma, sigma), y0=y0, t_span=[0, days], t_eval=t_eval, method='LSODA', atol=1e-4, rtol=1e-6)

            t  = ode_sol['t']
            S  = ode_sol['y'][0]
            E  = ode_sol['y'][1]
            I  = ode_sol['y'][2]
            R  = ode_sol['y'][3]
        else:
            # Integrate the SEIR equations with typical approach (dopri..)
            # Using the standard ODEint function
            ode_sol = odeint(seir_ode, y0, t_eval, args=(N, beta, gamma, sigma), atol=1e-4, rtol=1e-6)
            t = t_eval
            S, E, I, R = ode_sol.T

        # Pack timeseries
        sol_ode_timeseries = np.vstack((t, S, E, I, R))

    else:
        ''' Compartment structure of armed forces SEIR model
            N = S + E + I + R + D
        '''

        # Initial conditions vector
        S0, E0, I0, R0, D0 = y0

        if solver_type:
            # Integrate the SEIR equations with LSODA approach (adaptive)
            ode_sol = solve_ivp(lambda t, X: seir_deaths_ivp(t, X, N, beta, gamma, sigma, m), y0=y0, t_span=[0, days], t_eval=t_eval, method='LSODA')

            t   = ode_sol['t']
            S   = ode_sol['y'][0]
            E   = ode_sol['y'][1]
            I   = ode_sol['y'][2]
            Re  = ode_sol['y'][3]
            D   = ode_sol['y'][4]

        else:     
            # Integrate the SEIR equations with typical approach (dopri..)
            # Using the standard ODEint function
            ode_sol = odeint(seir_deaths_ode, y0, t_eval, args=(N, beta, gamma, sigma. m))
            t = t_eval
            S, E, I, Re, D = ode_sol.T
        
        # Pack timeseries            
        sol_ode_timeseries = np.vstack((t, S, E, I, Re, D))    

    return  sol_ode_timeseries  

###################################################
######## SEIR Model simulation Simulation ########
###################################################

if sim_num == 0:
        ''' Compartment structure of armed forces SEIR model
            N = S + E + I + R 
        '''        
        # Initial conditions vector
        S0 = N - E0 - I0 - R0 
        y0 = S0, E0, I0, R0
        print("S0=",S0, "E0=",E0, "I0=",I0, "R0=",R0)

        # Simulation Options
        seir_type   = 0 # SEIR no deaths
        solver_type = 1 # ivp - LSODA

else:
        ''' Compartment structure of armed forces SEIR model with deaths
            N = S + E + I + R + D
        '''
        S0 = N - E0 - I0 - R0 - D0
        y0 = S0, E0, I0, R0, D0
        print("S0=",S0, "E0=",E0, "I0=",I0, "R0=",R0, "D0", D0)

        # Simulation Options
        seir_type   = 1 # SEIR with deaths
        solver_type = 1 # ivp - LSODA

# Simulate ODE equations
sol_ode_timeseries = simulate_seirModel(seir_type, solver_type, y0, days, 1)

# Unpack timseries
if sim_num == 0:
    t = sol_ode_timeseries[0]    
    S = sol_ode_timeseries[1]    
    E = sol_ode_timeseries[2]    
    I = sol_ode_timeseries[3]    
    R = sol_ode_timeseries[4]    
else:
    t  = sol_ode_timeseries[0]    
    S  = sol_ode_timeseries[1]    
    E  = sol_ode_timeseries[2]    
    I  = sol_ode_timeseries[3]    
    Re = sol_ode_timeseries[4]    
    D  = sol_ode_timeseries[5]  
    R   = Re + D

# Accumulated Total Cases
T  = I + R

print("t=",  t[-1])
print("ST=", S[-1])
print("ET=", E[-1])
print("IT=", I[-1])
print("RT=", R[-1])
print("TT=", T[-1])

if sim_num > 0:
    print("DT=", D[-1])
    print("ReT=",Re[-1])


# Estimated Final epidemic size (analytic) not-dependent on simulation
init_guess   = 0.0001
r0_test      = r0
SinfN  = fsolve(epi_size, init_guess)
One_SinfN = 1 - SinfN
print('*****   Final Epidemic Size    *****')
print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

print('*****   Results    *****')
peak_inf_idx = np.argmax(I)
peak_inf     = I[peak_inf_idx]
print('Peak Instant. Infected = ', peak_inf,'by day=', peak_inf_idx)
peak_total_inf     = T[peak_inf_idx]
print('Total Cases when Peak = ', peak_total_inf,'by day=', peak_inf_idx)

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax1 = plt.subplots()
if sim_num > 0:
    txt_title_sim = r"COVID-19 Vanilla SEIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, m={m:1.3f})"
    fig.suptitle(txt_title_sim.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, m=m),fontsize=15)
else:
    txt_title_sim = r"COVID-19 Vanilla SEIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f})"
    fig.suptitle(txt_title_sim.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv),fontsize=15)    



# Variable evolution
ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
ax1.plot(t, E/N, 'm',   lw=2, label='Exposed')
ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')
ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')

if sim_num == 0:
    ax1.plot(t, R/N, 'g--',  lw=1,  label='Recovered')    
else:
    ax1.plot(t, Re/N, 'g--',  lw=1,  label='Recovered')
    ax1.plot(t, D/N, 'b--',  lw=1,  label='Dead')

# Plot Final Epidemic Size
ax1.plot(t, One_SinfN*np.ones(len(t)), 'm--')
txt1 = "{per:2.2f} infected"
ax1.text(t[0], One_SinfN - 0.05, txt1.format(per=One_SinfN[0]), fontsize=12, color='m')

# Plot peak points
ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)
ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
if sim_num < 2:
    txt_title = r"Peak infected: {peak_inf:5.0f} by day {peak_days:2.0f}" 
    txt_title2 = r"Total Cases: {peak_total:5.0f} by day {peak_days:2.0f}" 
else: 
    txt_title = r"Peak infected: {peak_inf:5.0f} by day {peak_days:2.0f} from March 21" 
    txt_title2 = r"Total Cases: {peak_total:5.0f} by day {peak_days:2.0f} from March 21" 
ax1.text(peak_inf_idx+10, peak_inf/N, txt_title.format(peak_inf=peak_inf, peak_days= peak_inf_idx), fontsize=12, color="r")
ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf, peak_days= peak_inf_idx), fontsize=12, color="r")


# Making things beautiful
ax1.set_xlabel('Time /days', fontsize=12)
ax1.set_ylabel('Percentage of Population', fontsize=12)
ax1.yaxis.set_tick_params(length=0)
ax1.xaxis.set_tick_params(length=0)
ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax1.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(True)

fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
fig.set_size_inches(20.5/2, 14.5/2, forward=True)


plt.savefig('./snaps/vanillaSEIR_timeEvolution_%i.png'%sim_num, bbox_inches='tight')
plt.savefig('./snaps/vanillaSEIR_timeEvolution_%i.pdf'%sim_num, bbox_inches='tight')

#################################################################
######## Plots Simulation with reproductive/growth rates ########
#################################################################
do_growth = 1
if do_growth:
    # Analytic growth rate
    effective_Rt = r0 * (S/N)
    growth_rates = gamma * (effective_Rt - 1)

    ####### Plots for Growth Rates #######
    fig, (ax1, ax2) = plt.subplots(1,2)
    if sim_num > 0:
        fig.suptitle(txt_title_sim.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, m=m),fontsize=15)
    else:
        fig.suptitle(txt_title_sim.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv),fontsize=15)    


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
    effRT_crossing = ids_less_1[0][1]
    ax1.plot(effRT_crossing, 1,'ro', markersize=12)
    ax1.text(effRT_crossing-10, 1-0.2,str(effRT_crossing), fontsize=10, color="r")
    ax1.set_ylabel('Rt (Effective Reproductive Rate)', fontsize=12)
    ax1.set_xlabel('Time[days]', fontsize=12)
    ax1.set_ylim(0,4)    

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
    rI_crossing = ids_less_1[0][1]
    ax2.plot(rI_crossing, 0,'ro', markersize=12)
    ax2.text(rI_crossing-10, 0-0.04,str(rI_crossing), fontsize=10, color="r")
    fig.set_size_inches(27.5/2, 12.5/2, forward=True)

    plt.savefig('./snaps/vanillaSIR_growthRates_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./snaps/vanillaSIR_growthRates_%i.pdf'%sim_num, bbox_inches='tight')


#############################################################
######## Dependence of R0 on Final Epidemic Behavior ########
#############################################################
# Final epidemic size (analytic)
# r0_vals     = np.linspace(1,5,100) 
# init_guess  = 0.0001
# Sinf_N      =   []
# Sinf_S0     =   []
# for ii in range(len(r0_vals)):
#     r0_test = r0_vals[ii]
#     Sinf_N.append(fsolve(epi_size, init_guess))     
#     Sinf_S0.append(1 - Sinf_N[ii])


# # Plots
# fig0, ax0 = plt.subplots()
# ax0.plot(r0_vals, Sinf_S0, 'r', lw=2, label='Susceptible')
# ax0.set_ylabel('$1 - S_{\infty}/N$ (percentage of population infected)', fontsize=12)
# ax0.set_xlabel('$R_0$', fontsize=12)

# # Current estimate of Covid R0
# plt.title('Final Size of Epidemic Dependence on $R_0$ estimate',fontsize=15)
# ax0.plot(r0, One_SinfN, 'ko', markersize=5, lw=2)

# # Plot mean
# txt = 'Covid R0({r0:3.3f})'
# ax0.text(r0 - 0.45, One_SinfN + 0.05,txt.format(r0=r0_test), fontsize=10)
# plt.plot([r0]*10,np.linspace(0,One_SinfN,10), color='black')
# txt = "{Sinf:3.3f} Infected"
# ax0.text(1.1, One_SinfN - 0.025,txt.format(Sinf=One_SinfN[0]), fontsize=8)
# plt.plot(np.linspace(1,[r0],10), [One_SinfN]*10, color='black')

# ax0.text(4, 0.75, r"${\cal R}_0 \equiv \frac{ \beta } {\gamma}$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
# fig0.set_size_inches(18.5/2, 12.5/2, forward=True)
# plt.savefig('./snaps/armedSIR_finalSize_%i.png'%sim_num, bbox_inches='tight')
# plt.savefig('./snaps/armedSIR_finalSize_%i.pdf'%sim_num, bbox_inches='tight')


plt.show()


