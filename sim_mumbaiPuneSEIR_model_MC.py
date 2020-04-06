import random
import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats
from   scipy.stats import gamma as gamma_dist
import scipy.special as sps
from scipy.stats import sem
from scipy.stats import t as tdist
from scipy import mean
confidence = 0.95
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


############################################
######## Parameters for Simulation ########
############################################

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
Q0           = 249               
T0           = 334               # This is the total number of comfirmed cases for March 21st, not used it seems?                                   

# Derived Model parameters and 
beta       = r0 / gamma_inv
sigma      = 1.0 / sigma_inv
gamma      = 1.0 / gamma_inv
tau_q      = 1.0 /tau_q_inv
# tau_q      = tau_q_inv

# Control variable:  percentage quarantined
q           = 0.01
# Q0 is 1% of total infectious; i.e. I0 + Q0 (as described in report)
# In the report table 1, they write number of Quarantined as SO rather than Q0
# Q0, is this a typo? 
# Number of Infectuos as described in report    
I0          = ((1-q)/(q)) * Q0  

# The initial number of exposed E(0) is not defined in report, how are they computed?
contact_rate = 10                     # number of contacts an individual has per day
E0           = (contact_rate - 1)*I0  # Estimated exposed based on contact rate and inital infected

######## Simulation Functions ########
######################################
# Equation to estimate final epidemic size (infected)
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

# The SEIR model differential equations with mortality rates
def seir_mumbai(t, X, N, beta, gamma, sigma, m, q, tau_q):
    S, E, I, Q, R, D = X

    # Original State equations for SEIR
    dSdt  = - (beta*S*I)/N 
    dEdt  = (beta*S*I)/N - sigma*E    

    # Incorporating Quarantine components
    dIdt  = sigma*E - gamma*I - q*I - m*I
    dQdt  = q*I - tau_q*Q - m*Q
    dRdt  = gamma*I + tau_q*Q
    dDdt  = m*I + m*Q 

    return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt



###################################################
######## SEIR Model simulation Simulation ########
###################################################

# Control variable:  percentage quarantined
q           = 0.40

# Model Parameters to sample from Gamma Distributions
gamma_inv, gamma_inv_shape = 7 , 0.1    # From report (mean, shape)
sigma_inv, sigma_inv_shape = 5.1, 0.1   # From report (mean, shape)
r0, r0_shape               = 2.28 , 0.1 # From report (mean, shape)    

gamma_calc_option = 1

if gamma_calc_option == 0:
    # Sample 1000 points from gamma distribution of gamma_inv
    # Trief this but its a hack to get some good numbers.. 
    # How to convert from (mean, shape) params --> (shape, scale) params ???
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.gamma.html
    gamma_inv_samples = gamma_inv*np.random.gamma(gamma_inv, gamma_inv_shape, 1000)
    sigma_inv_samples = sigma_inv*np.random.gamma(sigma_inv, sigma_inv_shape, 1000)
    r0_samples        = r0*np.random.gamma(r0, r0_shape, 1000)
else:
    # For gamma pdf plots
    x           = np.linspace(1E-6, 10, 1000)
    num_samples = 1000

    #### Gamma distributed samples for gamma_inv ####
    #### --> This might be wrong?!?!?
    k = 1
    loc = gamma_inv
    theta = gamma_inv_shape
    gamma_inv_dist    = gamma_dist(k, loc, theta)
    gamma_inv_samples = gamma_inv_dist.rvs(num_samples)

    # Plot gamma samples and pdf
    count, bins, ignored = plt.hist(gamma_inv_samples, 50, density=True)
    plt.plot(x, gamma_inv_dist.pdf(x), 'r',
             label=r'$k=%.1f,\ \theta=%.1f$' % (k, theta))

    #### Gamma distributed samples for sigma_inv ####
    k = 1
    loc = sigma_inv
    theta = sigma_inv_shape
    sigma_inv_dist = gamma_dist(k, loc, theta)
    sigma_inv_samples = sigma_inv_dist.rvs(num_samples)

    # Plot sigma samples and pdf
    plt.plot(x, sigma_inv_dist.pdf(x), 'g',
             label=r'$k=%.1f,\ \theta=%.1f$' % (k, theta))

    count, bins, ignored = plt.hist(sigma_inv_samples, 50, density=True)


    #### Gamma distributed samples for r0 ####
    k = 1
    loc = r0
    theta = r0_shape
    r0_dist = gamma_dist(k, loc, theta)
    r0_samples = r0_dist.rvs(num_samples)

    # Plot r0 samples and pdf
    plt.plot(x, r0_dist.pdf(x), 'b',
             label=r'$k=%.1f,\ \theta=%.1f$' % (k, theta))
    count, bins, ignored = plt.hist(r0_samples, 50, density=True)

    plt.show()

# A grid of time points (in days) for each simulation
t_eval = np.arange(0, days, 1)

simulations = 1000
S_sims  = np.empty(shape=(simulations,days))
E_sims  = np.empty(shape=(simulations,days))
I_sims  = np.empty(shape=(simulations,days))
Q_sims  = np.empty(shape=(simulations,days))
Re_sims = np.empty(shape=(simulations,days))
D_sims  = np.empty(shape=(simulations,days))
for ii in range(simulations):
    gamma_inv_sample = gamma_inv_samples[ii]
    sigma_inv_sample = sigma_inv_samples[ii]
    r0_sample = r0_samples[ii]

    ### TODO:  Put all of this in a class later
    # Derived Model parameters and 
    beta       = r0_sample / gamma_inv_sample
    sigma      = 1.0 / sigma_inv_sample
    gamma      = 1.0 / gamma_inv_sample

    print('*****   Hyper-parameters    *****')
    print('N=',N,'days=',days, 'r0=',r0_sample, 'gamma_inv (days) = ',gamma_inv_sample, 'tauq_inv (days) = ',tau_q_inv, 'sigma_inv (days) = ',sigma_inv_sample)

    print('*****   Model-parameters    *****')
    print('beta=',beta,'gamma=', gamma, 'sigma', sigma, 'tau_q', tau_q, 'm', m)

    ''' Compartment structure of armed forces SEIR model
        N = S + E + I + Q + R + D
    '''
    S0 = N - E0 - I0 - Q0 - R0 - D0
    print("S0=",S0)
    print("E0=",E0)
    print("I0=",I0)
    print("Q0=",D0)
    print("R0=",R0)
    print("D0=",D0)
    
    # Initial conditions vector
    y0 = S0, E0, I0, Q0, R0, D0

    # Integrate the SEIR equations over the time grid, with bet
    ode_sol = solve_ivp(lambda t, X: seir_mumbai(t, X, N, beta, gamma, sigma, m, q, tau_q), y0=y0, t_span=[0, days], t_eval=t_eval, method='LSODA')

    t              = ode_sol['t']
    S_sims[ii,:]   = ode_sol['y'][0]
    E_sims[ii,:]   = ode_sol['y'][1]
    I_sims[ii,:]   = ode_sol['y'][2]
    Q_sims[ii,:]   = ode_sol['y'][3]
    Re_sims[ii,:]  = ode_sol['y'][4]
    D_sims[ii,:]   = ode_sol['y'][5]


t = t_eval
# Compute mean and confidence interval of simulations
# Confidence intervals might be wrong, ankit said he saw negative numbers...
S = np.mean(S_sims,0)
E = np.mean(E_sims,0)

# Infected Cases
I = np.mean(I_sims,0)
I_std_err = sem(I_sims,0)
I_h = I_std_err * tdist.ppf((1 + confidence) / 2, days - 1)
I_up = I+I_h
I_low = I-I_h

# Quarantined Cases
Q = np.mean(Q_sims,0)
Q_std_err = sem(Q_sims,0)
Q_h   = Q_std_err * tdist.ppf((1 + confidence) / 2, days - 1)
Q_up  = Q+Q_h
Q_low = Q-Q_h

Re = np.mean(Re_sims,0)
D = np.mean(D_sims,0)

# Compute auxilliary variables
R   = Re + D + Q
T  = I + R 
TA = E + I + Q
Inf = I + Q
print("t=",  t[-1])
print("ST=", S[-1])
print("ET=", E[-1])
print("IT=", I[-1])
print("QT=", Q[-1])
print("InfT=", Inf[-1])

print("RT=", R[-1])
print("DT=", D[-1])
print("ReT=",Re[-1])

print("TT=", T[-1])
print("TAT=", TA[-1])


print('*****   Results    *****')
max_inf_idx = np.argmax(I)
max_inf     = I[max_inf_idx]
print('Peak Infected = ', max_inf,'by day=',max_inf_idx)

max_act_idx = np.argmax(Inf)
max_act     = Inf[max_act_idx]
print('Peak Infected+Quarantined = ', max_act, 'by day=',max_act_idx)

est_act     = Inf[65]
print('3Million Estimate Exp+Inf = ', est_act,'by day 65')

# Final epidemic size (analytic)
init_guess   = 0.0001
r0_test      = r0
covid_SinfN  = fsolve(epi_size, init_guess)
covid_1SinfN = 1 - covid_SinfN
print('*****   Final Epidemic Size    *****')
print('Covid r0 = ', r0_test, 'Covid Sinf/S0 = ', covid_SinfN[0], 'Covid Sinf/S0 = ', covid_1SinfN[0])

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax1 = plt.subplots()

gamma_inv = np.mean(gamma_inv_samples)
sigma_inv = np.mean(sigma_inv_samples)
r0        = np.mean(r0_samples)

txt_title = r"COVID-19 Mumbai SEIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, 1/$\tau_q$={tau_q_inv:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, tau_q_inv = tau_q_inv),fontsize=15)

plot_all = 0
plot_inf = 1

# Evolution of infected cases with confidence intervals
if plot_inf:
    ax1.plot(t, I/N, 'r',   lw=2,       label='Infected')
    ax1.plot(t, I_up/N, 'r--',   lw=2,   label='95 CI')
    ax1.plot(t, I_low/N, 'r--',   lw=2,   label='95 CI')
    ax1.plot(65, est_act/N,'ro', markersize=8)
    ax1.plot(t, Q/N, 'm',   lw=2,       label='Quarantined')
    ax1.plot(t, Q_up/N, 'm--',   lw=2,   label='95 CI')
    ax1.plot(t, Q_low/N, 'm--',   lw=2,   label='95 CI')
    ax1.plot(t, (Q+I)/N, 'k',   lw=2,       label='Infectuos (I+Q)')
    ax1.plot(t, (Q_up+I_up)/N, 'k--',   lw=2,   label='95 CI')
    ax1.plot(t, (Q_low+I_low)/N, 'k--',   lw=2,   label='95 CI')
    ax1.plot(t, D/N, 'b--',  lw=1,  label='Dead')


# Variable evolution
if plot_all:
    ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
    ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
    ax1.plot(t, Re/N, 'g--',  lw=1,  label='Recovered')
    ax1.plot(t, R/N, 'g',  lw=2,  label='Recovered+Dead+Quarantined')
    # Plot Final Epidemic Size
    ax1.plot(t, covid_1SinfN*np.ones(len(t)), 'm--')
    txt1 = "{per:2.2f} infected"
    ax1.text(t[0], covid_1SinfN - 0.05, txt1.format(per=covid_1SinfN[0]), fontsize=12, color='m')
    ax1.plot(t, E/N, 'm',   lw=2, label='Exposed')
    ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')
    ax1.plot(t, Inf/N, 'r--',   lw=2,   label='Infectious (I+Q)')
    ax1.plot(t, TA/N, 'c--', lw=1.5,   label='Exposed+Infected')
    ax1.plot(t, D/N, 'b--',  lw=1,  label='Dead')
    ax1.plot(t, Q/N, 'm--',  lw=1,  label='Quarantined')

# Plot peak points
ax1.plot(max_inf_idx, max_inf/N,'ro', markersize=8)
ax1.plot(max_act_idx, max_act/N,'ro', markersize=8)
ax1.plot(65, est_act/N,'ro', markersize=8)
txt_title = r"Peak infected: {peak_inf:5.0f} by day {peak_days:2.0f} from March 21" 
txt_title2 = r"Peak infected+quarantined: {peak_act:5.0f} by day {peak_days:2.0f} from March 21" 
txt_title3 = r"Est. inf: {est_act:5.0f} by day May 25" 
ax1.text(max_inf_idx+10, max_inf/N, txt_title.format(peak_inf=max_inf, peak_days= max_inf_idx), fontsize=10, color="r")
ax1.text(max_act_idx+10, max_act/N, txt_title2.format(peak_act=max_act, peak_days= max_act_idx), fontsize=10, color="r")
ax1.text(0, est_act/N, txt_title3.format(est_act=est_act), fontsize=10, color="r")

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

sim_num = 1
plt.savefig('./snaps/mumbaiSEIR_timeEvolution_%i.png'%sim_num, bbox_inches='tight')
plt.savefig('./snaps/mumbaiSEIR_timeEvolution_%i.pdf'%sim_num, bbox_inches='tight')

#################################################################
######## Plots Simulation with reproductive/growth rates ########
#################################################################
do_growth = 0
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
    ax1.text(t[-1]-100, 1 + 0.01, txt1.format(per=1), fontsize=12, color='r')
    ax1.text(t[-1]-100,2.5, r"${\cal R}_t \equiv \left( \frac{S (t) }{N (t) } \right) {\cal R}_0$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))


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
    txt_title = r"COVID-19 Mumbai SEIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f}, 1/$\tau_q$={tau_q_inv:1.3f})"
    fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, tau_q_inv = tau_q_inv),fontsize=15)

    # Plot of temporal growth rate
    ax2.plot(t, growth_rates, 'k', lw=2, label='rI (temporal growth rate)')
    ax2.text(t[0] + 0.02, growth_rates[0] - 0.02,r'${r}_I(t)$', fontsize=10)

    ax2.plot(t, 0*np.ones(len(t)), 'r-')
    txt1 = r"Critical ($r_I$={per:2.2f})"
    ax2.text(t[-1]-100, 0 + 0.01, txt1.format(per=0), fontsize=12, color='r')
    ax2.text(t[-1]-100, 0.2, r"$r_I  \equiv \gamma \left[ {\cal R}_t - 1 \right]$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
    ax2.text(t[-1]-100, 0.1, r"$\frac{ dI}{dt} = r_I \, I $", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
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

    plt.savefig('./snaps/mumbaiSIR_growthRates_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./snaps/mumbaiSIR_growthRates_%i.pdf'%sim_num, bbox_inches='tight')


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
# ax0.plot(r0, covid_1SinfN, 'ko', markersize=5, lw=2)

# # Plot mean
# txt = 'Covid R0({r0:3.3f})'
# ax0.text(r0 - 0.45, covid_1SinfN + 0.05,txt.format(r0=r0_test), fontsize=10)
# plt.plot([r0]*10,np.linspace(0,covid_1SinfN,10), color='black')
# txt = "{Sinf:3.3f} Infected"
# ax0.text(1.1, covid_1SinfN - 0.025,txt.format(Sinf=covid_1SinfN[0]), fontsize=8)
# plt.plot(np.linspace(1,[r0],10), [covid_1SinfN]*10, color='black')

# ax0.text(4, 0.75, r"${\cal R}_0 \equiv \frac{ \beta } {\gamma}$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
# fig0.set_size_inches(18.5/2, 12.5/2, forward=True)
# plt.savefig('./snaps/armedSIR_finalSize_%i.png'%sim_num, bbox_inches='tight')
# plt.savefig('./snaps/armedSIR_finalSize_%i.pdf'%sim_num, bbox_inches='tight')


plt.show()



