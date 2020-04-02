import numpy as np
from   scipy.integrate import odeint
from   scipy.optimize import fsolve
from   scipy import stats
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
r0         = 4 # Basic Reproductive Rate 2.65, estimates between 2.2-6.5
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

#############################################################
######## Dependence of R0 on Final Epidemic Behavior ########
#############################################################

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
ax0.set_ylabel('Sinf/S0 (percentage of population infected)')
ax0.set_xlabel('R0')

# Current estimate of Covid R0
plt.title('Final Size of Epidemic Dependence on R0 estimate',fontsize=15)
ax0.plot(r0_test, covid_SinfS0, 'ko', markersize=5, lw=2)

# Plot mean
txt = 'Covid R0({r0:3.3f})'
ax0.text(r0_test - 0.45, covid_SinfS0 + 0.05,txt.format(r0=r0_test), fontsize=10)
plt.plot([r0]*10,np.linspace(0,covid_SinfS0,10), color='black')
txt = "{Sinf:3.3f} Infected"
ax0.text(1.1, covid_SinfS0 - 0.025,txt.format(Sinf=covid_SinfS0[0]), fontsize=8)
plt.plot(np.linspace(1,[r0],10), [covid_SinfS0]*10, color='black')

# Sample R0 values from Gaussian distribution for Covid estimate
mu_r0, sigma_r0 = r0, 0.15  # mean and standard deviation
r0_gauss   = stats.norm.pdf(r0_vals,mu_r0,sigma_r0)
r0_samples = np.random.normal(mu_r0, sigma_r0, 25)
covid_SinfS0_samples  =   []
for ii in range(len(r0_samples)):
    r0_test = r0_samples[ii]
    covid_SinfS0_samples.append(1 - fsolve(epi_size, init_guess))

plt.plot(r0_samples, covid_SinfS0_samples, 'ko', markersize=5, lw=2, alpha=0.5)

# Plot +std
r0_test          = r0+2*sigma_r0
covid_SinfS0_plus = 1 - fsolve(epi_size, init_guess)
ax0.text(r0_test + 0.02, covid_SinfS0_plus - 0.15,r'$R_0+2\sigma$', fontsize=10)
txt = '$\sigma$=' + str(sigma_r0)
ax0.text(r0_test + 0.02, covid_SinfS0_plus - 0.5,txt, fontsize=10)
plt.plot([r0_test]*10,np.linspace(0,covid_SinfS0_plus,10), color='black', alpha=0.5)
txt = "{Sinf:3.3f} Infected"
ax0.text(1.1, covid_SinfS0_plus + 0.01,txt.format(Sinf=covid_SinfS0_plus[0]), fontsize=8)
plt.plot(np.linspace(1,[r0_test],10), [covid_SinfS0_plus]*10, color='black', alpha=0.5)


# Plot -std
r0_test          = r0-2*sigma_r0
covid_SinfS0_plus = 1 - fsolve(epi_size, init_guess)
ax0.text(r0_test - 0.25, covid_SinfS0_plus - 0.15,r'$R_0-2\sigma$', fontsize=10)
plt.plot([r0_test]*10,np.linspace(0,covid_SinfS0_plus,10), color='black', alpha=0.5)
txt = "{Sinf:3.3f} Infected"
ax0.text(1.1, covid_SinfS0_plus - 0.05,txt.format(Sinf=covid_SinfS0_plus[0]), fontsize=8)
plt.plot(np.linspace(1,[r0_test],10), [covid_SinfS0_plus]*10, color='black', alpha=0.5)

ax0.xaxis.set_tick_params(length=0)
ax0.yaxis.set_tick_params(length=0)

# Plot Gaussian and sampled from R0
ax01       = ax0.twinx()   # instantiate a second axes that shares the same x-axis
plt.plot(r0_vals,r0_gauss/mu_r0, color='black', alpha=0.5)
plt.plot(r0_samples, [0.15]*len(r0_samples), 'ko', markersize=5, lw=2, alpha=0.5)
plt.ylim(0.1,5)
ax01.xaxis.set_tick_params(length=0)
ax01.yaxis.set_tick_params(length=0)


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

# Analytic growth rate
effective_Rt = r0 * (S/N)
growth_rates = gamma * (effective_Rt - 1)

print('*****   Results    *****')
max_inf_idx = np.argmax(I)
max_inf     = I[max_inf_idx]
print('Turning point Infected = ', max_inf,'by day=',max_inf_idx)

##################################
######## Plots Generation ########
##################################
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, (ax1, ax2) = plt.subplots(1,2)

txt_title = "COVID-19 Vanilla SIR Model Dynamics (N={N:2.0f},R0={R0:1.3f},1/gamma={gamma_inv:1.3f}, beta={beta:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=r0, gamma_inv = gamma_inv, beta= beta),fontsize=15)

# Variable evolution
ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
ax1.plot(t, I/N, 'r', lw=2, label='Infected')
ax1.plot(t, R/N, 'g', lw=2, label='Recovered')

# Inflection point
ax1.plot(max_inf_idx, max_inf/N, 'md', markersize=12, lw=2)
txt = "Tipping point @ {day:3.0f} days "
ax1.text(max_inf_idx + 5, max_inf/N - 0.05,txt.format(day=max_inf_idx), fontsize=12)
ax1.plot(t, covid_SinfS0*np.ones(len(t)), 'm--')
txt1 = "{per:2.2f} population infected"
ax1.text(t[0], covid_SinfS0 - 0.05, txt1.format(per=covid_SinfS0[0]), fontsize=12, color='m')


ax1.set_xlabel('Time /days')
ax1.set_ylabel('Percentage of Population')
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


plt.show()



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

# fig, ax = plt.subplots()
# ax.plot(S, I, 'k', lw=2, label='Susceptible')
# ax.set_xlabel('S (Susceptible)')
# ax.set_ylabel('I (Infected)')
# plt.title('COVID-19 SIR Model Phase-Plane',fontsize=15)