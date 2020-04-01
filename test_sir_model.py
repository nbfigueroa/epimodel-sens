import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Total population, N.
N    = 1400000000 # China population
days = 90

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# Contact rate (beta), and mean recovery rate (gamma) (in 1/days).
# Estimtated values
infperiod  = 4.5  # SEIRS dudes: 6.8, Luis B.: 4.5, 1/0.3253912
r0         = 2.85 # Basic Reproductive Rate 2.65, estimates between 2.2-6.5
beta       = r0 / infperiod
gamma      = 1.0 / infperiod

print('beta=',beta,'gamma=',gamma, 'r0=',r0, 'infperiod=',infperiod)

# A grid of time points (in days)
t = np.linspace(0, days, days)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta/N * S * I 
    dIdt = beta/N * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ode_sol = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ode_sol.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('COVID-19 SIR Model Predictions (N=1.4e9)',fontsize=15)
ax1.plot(t, S, 'k', lw=2, label='Susceptible')
ax1.plot(t, I, 'r', lw=2, label='Infected')
ax1.plot(t, R, 'g', lw=2, label='Recovered')
ax1.set_xlabel('Time /days')
ax1.set_ylabel('Number (1 person)')
ax1.yaxis.set_tick_params(length=0)
ax1.xaxis.set_tick_params(length=0)
ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax1.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(True)

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


fig, ax = plt.subplots()
ax.plot(S, I, 'k', lw=2, label='Susceptible')
ax.set_xlabel('S (Susceptible)')
ax.set_ylabel('I (Infected)')
plt.title('COVID-19 SIR Model Phase-Plane',fontsize=15)

plt.show()