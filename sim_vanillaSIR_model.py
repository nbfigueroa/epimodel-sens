import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


############################################
######## Parameters for Simulation ########
############################################
N    = 1000000 
days = 90

# Contact rate (beta), and mean recovery rate (gamma) (in 1/days).
# Estimtated values
gamma_inv  = 4.5  # SEIRS dudes: 6.8, Luis B.: 4.5, 1/0.3253912
r0         = 2.25 # Basic Reproductive Rate 2.65, estimates between 2.2-6.5
beta       = r0 / gamma_inv
gamma      = 1.0 / gamma_inv

print('*****   Hyper-parameters    *****')
print('N=',N,'days=',days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)

print('*****   Model-parameters    *****')
print('beta=',beta,'gamma=',gamma)

######################################
######## Simulation Functions ########
######################################
# Equation to estimate final epidemic size (infected)
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta/N * S * I 
    dIdt = beta/N * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


###################################################
######## Initial Parameters for Simulation ########
###################################################

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# A grid of time points (in days)
t = np.linspace(0, days, days)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ode_sol = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ode_sol.T

max_inf_idx = np.argmax(I)
max_inf     = I[max_inf_idx]

# Final epidemic size (analytic)
r0_vals     = np.linspace(1,5,25) 
init_guess  = 0.0001
Sinf_N      =   []
Sinf_S0     =   []
for ii in range(len(r0_vals)):
	r0_test = r0_vals[ii]
	Sinf_N.append(fsolve(epi_size, init_guess))		
	Sinf_S0.append(1 - Sinf_N[ii])



fig0, ax0 = plt.subplots()
ax0.plot(r0_vals, Sinf_S0, 'r', lw=2, label='Sinf/S0')
ax0.set_ylabel('Sinf/S0 (percentage of population infected)')
ax0.set_xlabel('R0')
plt.title('Final Size of Epidemic Dependence on R0 estimate',fontsize=15)

r0_test      = 2.65
covid_SinfN  = fsolve(epi_size, init_guess)
covid_SinfS0 = 1 - covid_SinfN
print('Covid r0 = ', r0_test, 'Covid Sinf/S0 = ', covid_SinfN, 'Covid Sinf/S0 = ', covid_SinfS0) 

# Analytic growth rate
effective_Rt = r0 * (S/N)
growth_rates = gamma * (effective_Rt - 1)

print('*****   Results    *****')
print('Maximum infected = ', max_inf,'by day=',max_inf_idx)


##################################
######## Plots Generation ########
##################################
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, (ax1, ax2) = plt.subplots(1,2)

txt_title = "COVID-19 SIR Model Dynamics (N={N:2.0f},R0={R0:1.3f},1/gamma={gamma_inv:1.3f}, beta={beta:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=r0, gamma_inv = gamma_inv, beta= beta),fontsize=15)

# Variable evolution
ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
ax1.plot(t, I/N, 'r', lw=2, label='Infected')
ax1.plot(t, R/N, 'g', lw=2, label='Recovered')

# Max infected
ax1.plot(max_inf_idx, max_inf/N, 'md', markersize=12, lw=2)
txt = "Max infected ({max:5.0f}) @ {day:3.0f} days "
ax1.text(max_inf_idx + 5, max_inf/N - 0.05,txt.format(max=max_inf, day=max_inf_idx), fontsize=12)

ax1.set_xlabel('Time /days')
ax1.set_ylabel('Percentage of Population')
percs = max_inf/N*np.ones(len(t))
ax1.plot(t, percs, 'k--')
txt1 = "{per:2.2f}"
ax1.text(t[0], max_inf/N - 0.05,txt1.format(per=max_inf/N), fontsize=12)

ax1.yaxis.set_tick_params(length=0)
ax1.xaxis.set_tick_params(length=0)
ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax1.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(True)


# Plot curves in log-scale
ax2.plot(t, S, 'k', lw=2, label='Susceptible')
ax2.plot(t, I, 'r', lw=2, label='Infected')
ax2.plot(t, R, 'g', lw=2, label='Recovered')
plt.yscale('symlog', linthreshy=0.015)
ax2.set_xlabel('Time /days')
ax2.set_ylabel('Number (1 person)')
ax2.yaxis.set_tick_params(length=0)
ax2.xaxis.set_tick_params(length=0)
ax2.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax2.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax2.spines[spine].set_visible(True)

# Plot Grow Rate
# fig, (ax1, ax2) = plt.subplots(1,2)
# txt_title = "COVID-19 SIR Model Dynamics (N={N:2.0f},R0={R0:1.3f},1/gamma={gamma_inv:1.3f}, beta={beta:1.3f})"
# fig.suptitle(txt_title.format(N=N, R0=r0, gamma_inv = gamma_inv, beta= beta),fontsize=15)
# ax1.plot(t, growth_rates, 'k', lw=2, label='rI (temporal grow rate)')
# ax1.set_ylabel('rI (temporal grow rate)')
# ax1.set_xlabel('t')

# ax2.plot(t, effective_Rt, 'k', lw=2, label='Rt (Effective Reproductive Rate)')
# ax2.set_ylabel('Rt (Effective Reproductive Rate)')
# ax2.set_xlabel('t')



plt.show()



# fig, ax = plt.subplots()
# ax.plot(S, I, 'k', lw=2, label='Susceptible')
# ax.set_xlabel('S (Susceptible)')
# ax.set_ylabel('I (Infected)')
# plt.title('COVID-19 SIR Model Phase-Plane',fontsize=15)