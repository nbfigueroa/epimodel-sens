import numpy as np
from seirsplus.models import *
import matplotlib.pyplot as plt
from matplotlib import rc
from   scipy.optimize import fsolve

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


# Equation to estimate final epidemic size; i.e. (1 - S(inf))
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

sim_num = 1

if sim_num == 1:
    N            = 1486947036
    N            = 1375947036
    days         = 356
    gamma_inv    = 7  
    sigma_inv    = 5.1
    m            = 0.0043
    r0           = 2.28      
    
    # Initial values from March 21st "armed forces predictions"
    R0           = 23
    D0           = 5 
    T0           = 334  
    Q0           = 249     #Q0 is 1% of infectious I0
    I0           = (1.01/0.01) * Q0
    contact_rate = 10     # number of contacts per day
    E0           = (contact_rate - 1)*I0
    
    
    # Derived Model parameters and Control variable 
    beta       = r0 / gamma_inv
    sigma      = 1.0 / sigma_inv
    gamma      = 1.0 / gamma_inv
    model = SEIRSModel(beta=beta, sigma=sigma, gamma=gamma, initI=I0, initR=R0, initE=E0, initF=D0, initN=N, mu_I= m)
    model.run(T=days, dt=1, verbose=True)

    t = model.tseries
    S = model.numS
    E = model.numE
    I = model.numI
    Re = model.numR
    D  = model.numF
    R  = Re + D

# Simulation for a toy problem for implementation comparison
if sim_num == 2:
    # Simulation parameters
    S0   = 9990
    I0   = 1
    R0   = 0
    E0   = 9
    N    = S0 + I0 + R0 + E0 
    days = 50
    
    # Model parameters
    contact_rate = 10                     # number of contacts per day
    transmission_probability = 0.07       # transmission probability
    gamma_inv = 5                 # infectious period
    sigma_inv = 2                     # latent period 
    
    # Derived Model parameters and Control variable 
    beta   = contact_rate * transmission_probability
    gamma  = 1 / gamma_inv
    sigma  = 1 / sigma_inv
    r0     = beta / gamma

    model = SEIRSModel(beta=beta, sigma=sigma, gamma=gamma, initI=I0, initR=R0, initE=E0,  initN=N)
    model.run(T=50, dt=1, verbose=True)

    t  = model.tseries
    S  = model.numS
    E  = model.numE
    I  = model.numI
    R  = model.numR

T  = E + I + R
TA = E + I
print("t=",  t[-1])
print("ST=", S[-1])
print("ET=", E[-1])
print("IT=", I[-1])
print("RT=", R[-1])
print("TT=", T[-1])
print("TAT=", TA[-1])

if sim_num == 1:
    print("DT=", D[-1])
    print("ReT=",Re[-1])

print('*****   Results    *****')
max_inf_idx = np.argmax(I)
max_inf     = I[max_inf_idx]
print('Peak Infected = ', max_inf,'by day=',max_inf_idx)

max_act_idx = np.argmax(TA)
max_act     = TA[max_act_idx]
print('Peak Exp+Inf = ', max_inf,'by day=',max_act_idx)

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

txt_title = r"COVID-19 Vanilla SEIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}, 1/$\sigma$={sigma_inv:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv),fontsize=15)

# Variable evolution
if plot_all:
    ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
    ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
    ax1.plot(t, Re/N, 'g--',  lw=1,  label='Recovered')
    # Plot Final Epidemic Size
    ax1.plot(t, covid_1SinfN*np.ones(len(t)), 'm--')
    txt1 = "{per:2.2f} infected"
    ax1.text(t[0], covid_1SinfN - 0.05, txt1.format(per=covid_1SinfN[0]), fontsize=12, color='m')

ax1.plot(t, E/N, 'm',   lw=2, label='Exposed')
ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')
ax1.plot(t, TA/N, 'c--', lw=1.5,   label='Exposed+Infected')
ax1.plot(t, D/N, 'b--',  lw=1,  label='Dead')


# Plot peak points
ax1.plot(max_inf_idx, max_inf/N,'ro', markersize=8)
ax1.plot(max_act_idx, max_act/N,'ro', markersize=8)
if sim_num == 2:
    txt_title = r"Peak infected: {peak_inf:5.0f} by day {peak_days:2.0f}" 
    txt_title2 = r"Peak inf+exp: {peak_act:5.0f} by day {peak_days:2.0f}" 
else: 
    txt_title = r"Peak infected: {peak_inf:5.0f} by day {peak_days:2.0f} from March 21" 
    txt_title2 = r"Peak inf+exp: {peak_act:5.0f} by day {peak_days:2.0f} from March 21" 
ax1.text(max_inf_idx+10, max_inf/N, txt_title.format(peak_inf=max_inf, peak_days= max_inf_idx), fontsize=10, color="r")
ax1.text(max_act_idx+10, max_act/N, txt_title2.format(peak_act=max_act, peak_days= max_act_idx), fontsize=10, color="r")


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
fig.set_size_inches(18.5/2, 12.5/2, forward=True)

plt.savefig('vanillaSEIR_timeEvolution_%i.png'%sim_num, bbox_inches='tight')
plt.savefig('vanillaSEIR_timeEvolution_%i.pdf'%sim_num, bbox_inches='tight')

plt.show()


