import numpy as np
from   scipy.integrate import odeint
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Select simulation number
sim_num = 1

############################################
######## Parameters for Simulation ########
############################################
N    = 1000000
days = 180

# Contact rate (beta), and mean recovery rate (gamma) (in 1/days).
gamma_inv  = 4.5  # SEIRS dudes: 6.8, Luis B.: 4.5, 1/0.3253912 (Dave=10)

# Estimtated values
if sim_num == 1:
    beta_matrix = np.matrix([[0.35, 0.18, 0.11],[0.18, 0.27, 0.11],[0.11, 0.11, 0.30]])
    S_vector    = np.array([0.38, 0.31, 0.28])
    I_vector    = np.array([1e-5, 5e-5, 1e-5])
    r0_matrix   = gamma_inv * beta_matrix
    
if sim_num == 2:
    Beta        = np.matrix([[0.35, 0.18, 0.0],[0.18, 0.27, 0.0],[0.0, 0.0, 0.30]])
    S_vector    = np.array([0.38, 0.31, 0.28])
    I_vector    = np.array([1e-5, 5e-5, 1e-5])
    r0_matrix   = gamma_inv * beta_matrix

R_vector    = np.array([0, 0, 0])
print('*****   Model-parameters    *****')
print ('Beta=',beta_matrix)
print ('S_0=',S_vector)
print ('I_0=',I_vector)
print ('R_0=',R_vector)
print ('R0=', r0_matrix)


######################################
######## Simulation Functions ########
######################################
# Equation to estimate final epidemic size (infected)
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

# The SIR model differential equations.
def deriv(y_vector, t, N, beta_matrix, gamma):
    S, I, R = y
    dSdt = -beta/N * S * I 
    dIdt = beta/N * S * I  - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

###################################################
######## Initial Parameters for Simulation ########
###################################################
# Initial number of infected and recovered and susceptible individuals
S0 = N*S_0
I0 = N*I_0

# A grid of time points (in days)
t = np.linspace(0, days, days)

# Initial conditions matrix
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, with beta_matrix
ode_sol = odeint(deriv, y0, t, args=(N, beta_matrix, gamma))
S, I, R = ode_sol.T
T = I + R

print('*****   Results    *****')
max_inf_idx = np.argmax(I)
max_inf     = I[max_inf_idx]
print('Turning point Infected = ', max_inf,'by day=',max_inf_idx)

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax1 = plt.subplots()

txt_title = r"COVID-19 Vanilla SIR Model Dynamics (N={N:2.0f},1/$\gamma$={gamma_inv:1.3f}, $\beta$={beta:1.3f},$R_0$={R0:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=r0, gamma_inv = gamma_inv, beta= beta),fontsize=15)

# Variable evolution
ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
ax1.plot(t, I/N, 'r', lw=2,   label='Infected')
ax1.plot(t, R/N, 'g',  lw=2,  label='Recovered')
ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')


ax1.plot(t, covid_SinfS0*np.ones(len(t)), 'm--')
txt1 = "{per:2.2f} infected"
ax1.text(t[0], covid_SinfS0 - 0.05, txt1.format(per=covid_SinfS0[0]), fontsize=12, color='m')

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

plt.savefig('./snaps/subPopVanillaSIR_timeEvolution_%i.png'%sim_num, bbox_inches='tight')
plt.savefig('./snaps/subPopVanillaSIR_timeEvolution_%i.pdf'%sim_num, bbox_inches='tight')


#############################################################
######## Dependence of R0 on Final Epidemic Behavior ########
#############################################################

if false:
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
    ax0.text(4, 0.65, r"$\beta \sim {\cal N}(\mu_{\beta},(\sigma_{\beta}^2)$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
    fig0.set_size_inches(18.5/2, 12.5/2, forward=True)
    plt.savefig('./snaps/vanillaSIR_finalSize_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./snaps/vanillaSIR_finalSize_%i.pdf'%sim_num, bbox_inches='tight')



#################################################################
######## Plots Simulation with reproductive/growth rates ########
#################################################################
if false:
    r0_mean    = gamma_inv*beta
    # Analytic growth rate
    effective_Rt = r0_mean * (S/N)
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

    plt.savefig('./snaps/subPopVanillaSIR_growthRates_%i.png'%sim_num, bbox_inches='tight')
    plt.savefig('./snaps/subPopVanillaSIR_growthRates_%i.pdf'%sim_num, bbox_inches='tight')

plt.show()



