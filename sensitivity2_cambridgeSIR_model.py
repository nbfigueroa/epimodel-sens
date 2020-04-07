from matplotlib import rc
import pyross
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats

from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


# Equation to estimate final epidemic size (infected)
def epi_size(x):
    return np.log(x) + r0_test*(1-x)

# Plotting
x_axis_offset = 50

######################################################
####  Age group partitioning for Indian population  ##
######################################################

M = 16  # number of age groups

# load age structure data
my_data = np.genfromtxt('data/age_structures/India-2019.csv', delimiter=',', skip_header=1)
aM, aF = my_data[:, 1], my_data[:, 2]

# set age groups
Ni=aM+aF;   Ni=Ni[0:M];  N=np.sum(Ni)


################################################
####  Contact Matrices for Indian population  ##
################################################

# contact matrices
my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_home_1.xlsx', sheet_name='India',index_col=None)
CH = np.array(my_data)
my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_work_1.xlsx', sheet_name='India',index_col=None)
CW = np.array(my_data)
my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_school_1.xlsx', sheet_name='India',index_col=None)
CS = np.array(my_data)
my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_other_locations_1.xlsx', sheet_name='India',index_col=None)
CO = np.array(my_data)
my_data = pd.read_excel('data/contact_matrices_152_countries/MUestimates_all_locations_1.xlsx', sheet_name='India',index_col=None)
CA = np.array(my_data)

# matrix of total contacts
C=CH+CW+CS+CO


####################################
####  Fixed SIR Model Paramaters  ##
####################################

gIa   = 1./7            # recovery rate of asymptomatic infectives 
gIs   = 1./7            # recovery rate of symptomatic infectives 
alpha = 0.              # fraction of asymptomatic infectives 
fsa   = 1               # the self-isolation parameter   
gamma =  1/gIs       

# initial conditions    
Is_0 = np.zeros((M));  Is_0[4:11]=4;  Is_0[1:4]=1
Ia_0 = np.zeros((M))
R_0  = np.zeros((M))
S_0  = Ni - (Ia_0 + Is_0 + R_0)

beta  = 0.01566         # infection rate 

# Variables for beta distribution
error_perc = 10
beta_error = error_perc/100
beta_mean  = beta
beta_sigma = beta*(beta_error)
beta_plus  = beta_mean + 2*beta_sigma
beta_minus = beta_mean - 2*beta_sigma


beta_samples = [beta_mean, beta_plus, beta_minus]
S_samples = np.empty([3, 365])
I_samples = np.empty([3, 365])
R_samples = np.empty([3, 365])

for ii in range(3):
    beta = beta_samples[ii]
    # matrix for linearised dynamics
    L0 = np.zeros((M, M))
    L  = np.zeros((2*M, 2*M))

    for i in range(M):
        for j in range(M):
            L0[i,j]=C[i,j]*Ni[i]/Ni[j]

    L[0:M, 0:M]     =    alpha*beta/gIs*L0
    L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
    L[M:2*M, 0:M]   =    ((1-alpha)*beta/gIs)*L0
    L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0


    r0 = np.max(np.linalg.eigvals(L))
    print("The basic reproductive ratio for these parameters is", r0)


    ##############################
    ####  SIR Model Simulation  ##
    ##############################
    # duration of simulation and data file
    # Tf=21*2-1;  
    Nf=2000; filename='this.mat'

    # # the contact structure is independent of time 
    # def contactMatrix(t):
    #     return C

    # intantiate model
    parameters = {'alpha':alpha,'beta':beta, 'gIa':gIa,'gIs':gIs,'fsa':fsa}
    model = pyross.models.SIR(parameters, M, Ni)

    ###############################################
    ### Now run the real simulation   #
    ###############################################

    C=CH+CW+CS+CO
    Tf=365; 

    # matrix for linearised dynamics
    L0 = np.zeros((M, M))
    L  = np.zeros((2*M, 2*M))

    for i in range(M):
        for j in range(M):
            L0[i,j]=C[i,j]*Ni[i]/Ni[j]
    L[0:M, 0:M]     =    alpha*beta/gIs*L0
    L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
    L[M:2*M, 0:M]   =    ((1-alpha)*beta/gIs)*L0
    L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0

    r0 = np.max(np.linalg.eigvals(L))
    print("The basic reproductive ratio for these parameters is", r0)

    # initial conditions    
    Is_0 = np.zeros((M));  Is_0[4:11]=4;  Is_0[1:4]=1
    Ia_0 = np.zeros((M))
    R_0  = np.zeros((M))
    S_0  = Ni - (Ia_0 + Is_0 + R_0)

    S0 = S_0
    R0 = R_0
    I0 = Ia_0 + Is_0
    print("S0=",S0)
    print("I0=",I0)
    print("R0=",R0)

    scenario = 1
    if scenario == 0:
        def contactMatrix(t):
            return C
    if scenario == 1:
        # the contact matrix is time-dependent
        def contactMatrix(t):
            if t<21:
                xx = C
            elif 21<=t<42:
                xx = CH
            else:
                xx = C
            return xx
    if scenario == 2:
        def contactMatrix(t):
            if t<21:
                xx = C
            elif 21<=t<70:
                xx = CH
            else:
                xx = CH
            return xx



    ############ start simulation ############
    Nf=Tf; filename='this.mat'
    model.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf, filename)

    ############ Unpacking the simulated variables ############
    data=loadmat(filename)
    IC  = np.zeros((Nf))
    SC  = np.zeros((Nf))

    for i in range(M):
            IC += data['X'][:,2*M+i] 
            SC += data['X'][:,0*M+i]

    I = IC
    S = SC
    t = data['t'][0]
    R = N - S - I
    T = I + R

    S_samples[ii,:] = S
    I_samples[ii,:] = I
    R_samples[ii,:] = R


    # A grid of time points (in simulation_time)
    t = np.arange(0, 365, 1)

    gamma_inv = gIs

    print("t=",  t[-1])
    print("ST=", S[-1])
    print("IT=", I[-1])
    print("RT=", R[-1])
    print("TT=", T[-1])

    # Estimated Final epidemic size (analytic) not-dependent on simulation
    init_guess   = 0.0001
    r0_test      = float(r0)
    SinfN  = fsolve(epi_size, init_guess)
    One_SinfN = 1 - SinfN
    print('*****   Final Epidemic Size    *****')
    print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

    print('*****   Results    *****')
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

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()

    txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0^e$={R0:1.3f}, $\beta_e$={beta:1.3f}, 1/$\gamma$={gamma:1.3f})"
    fig.suptitle(txt_title.format(N=N, R0=float(r0), beta= beta, gamma = 1/gamma_inv),fontsize=20)

    # Variable evolution
    plot_all = 1
    if plot_all == 1:
        # ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
        ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
        # ax1.plot(t, R/N, 'g--',  lw=1,  label='Recovered')
# 
        # Plot Final Epidemic Size
        # ax1.plot(t, One_SinfN*np.ones(len(t)), 'm--')
        # txt1 = "Final Epidemic size (no intervention): 1-S(inf)/N={per:2.2f} percentage (analytic)"
        # ax1.text(t[-1]-200, One_SinfN + 0.02, txt1.format(per=One_SinfN[0]), fontsize=20, color='m')

        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.2f} million total cases as $t(end)$."
        ax1.text(t[0], (total_cases/N) - 0.05, txt1.format(per=total_cases/1000000), fontsize=20, color='r')


    ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')

    # Plot peak points
    ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)

    scale = 1000000
    txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} from March 4" 
    txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} from March 4" 
    ax1.text(peak_inf_idx+10, peak_inf/N, txt_title.format(peak_inf=peak_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if plot_all == 1:
        ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
        ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

    # Making things beautiful
    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax1.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    plt.savefig('./snaps/cambridgeSIR_timeEvolution_%i.png'%ii, bbox_inches='tight')
    plt.savefig('./snaps/cambridgeSIR_timeEvolution_%i.pdf'%ii, bbox_inches='tight')


#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
beta  = 0.01566         # infection rate 

print(S_samples)
# Unpack
S       = S_samples[0,:]
S_plus  = S_samples[1,:]
S_minus = S_samples[2,:]

I       = I_samples[0,:]
I_plus  = I_samples[1,:]
I_minus = I_samples[2,:]

R       = R_samples[0,:]
R_plus  = R_samples[1,:]
R_minus = R_samples[2,:]


T = I + R
T_minus = I_minus+ R_minus
T_plus = I_plus+ R_plus

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig, ax1 = plt.subplots()


txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0^e$={R0:1.3f}, $\beta_e$={beta:1.3f}, 1/$\gamma$={gamma:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=r0, gamma = 1/gamma_inv, beta= beta),fontsize=20)

# Variable evolution
ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.25)
ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.25)

ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
ax1.plot(t, I/N, 'r', lw=2,   label='Infected')
ax1.plot(t, I_minus/N, 'r--', lw=2, alpha=0.25)

ax1.plot(t, R_plus/N, 'g--',  lw=2, alpha=0.25)
# ax1.plot(t, R/N, 'g',  lw=2,  label='Recovered')
ax1.plot(t, R_minus/N, 'g--',  lw=2, alpha=0.25)

ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)

total_cases     = T[-1]
print('Total Cases when growth linear = ', total_cases)
ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
txt1 = "{per:2.2f} million total cases as $t(end)$."
ax1.text(t[0], (total_cases/N) - 0.05, txt1.format(per=total_cases/1000000), fontsize=20, color='r')

total_cases     = T_minus[-1]
print('Total Cases when growth linear = ', total_cases)
ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
txt1 = "{per:2.2f} million total cases as $t(end)$."
ax1.text(t[0], (total_cases/N) - 0.05, txt1.format(per=total_cases/1000000), fontsize=20, color='r')

total_cases     = T_plus[-1]
print('Total Cases when growth linear = ', total_cases)
ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
txt1 = "{per:2.2f} million total cases as $t(end)$."
ax1.text(t[0], (total_cases/N) - 0.05, txt1.format(per=total_cases/1000000), fontsize=20, color='r')
fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)

# Estimated Final epidemic size (analytic) not-dependent on simulation
init_guess   = 0.0001
r0_test      = float(r0)
SinfN  = fsolve(epi_size, init_guess)
One_SinfN = 1 - SinfN
print('*****   Final Epidemic Size    *****')
print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

print('*****   Results    *****')
peak_inf_idx =  np.argmax(I)
peak_inf     = I[peak_inf_idx]
print('Peak Instant. Infected = ', peak_inf,'by day=', peak_inf_idx)

peak_total_inf  = T[peak_inf_idx]
print('Total Cases when Peak = ', peak_total_inf,'by day=', peak_inf_idx)

total_cases     = T[-1]
print('Total Cases when growth linear = ', total_cases)


peak_inf_plus_idx =  np.argmax(I_plus)
peak_inf_plus     = I_plus[peak_inf_plus_idx]
print('Peak Instant. Infected - Error= ', peak_inf_plus,'by day=', peak_inf_plus_idx)

peak_inf_minus_idx =  np.argmax(I_minus)
peak_inf_minus     = I_plus[peak_inf_minus_idx]
print('Peak Instant. Infected + Error= ', peak_inf_minus,'by day=', peak_inf_minus_idx)

# Plot peak points
ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)
# Plot peak points
ax1.plot(peak_inf_plus_idx, peak_inf_plus/N,'ro', markersize=8)
# Plot peak points
ax1.plot(peak_inf_minus_idx, peak_inf_minus/N,'ro', markersize=8)


scale = 1000000
txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} from March 4" 
ax1.text(peak_inf_idx+10, peak_inf/N, txt_title.format(peak_inf=peak_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

# txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} from March 4" 
# ax1.text(peak_inf_plus_idx+10, peak_inf_plus/N, txt_title.format(peak_inf=peak_inf_plus/scale, peak_days= peak_inf_plus_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

# txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} from March 4" 
# ax1.text(peak_inf_minus_idx+10, peak_inf_minus/N, txt_title.format(peak_inf=peak_inf_minus/scale, peak_days= peak_inf_minus_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

if plot_all == 1:
    ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
    txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} from March 4" 
    ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

# ax1.plot(t, covid_SinfS0*np.ones(len(t)), 'm--')
# txt1 = "{per:2.2f} population infected"
# ax1.text(t[0], covid_SinfS0 - 0.05, txt1.format(per=covid_SinfS0[0]), fontsize=20, color='m')

ax1.set_xlabel('Time /days', fontsize=20)
ax1.set_ylabel('Fraction of Population', fontsize=20)
ax1.yaxis.set_tick_params(length=0)
ax1.xaxis.set_tick_params(length=0)
ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax1.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax1.spines[spine].set_visible(True)

fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
fig.set_size_inches(27.5/2, 16.5/2, forward=True)

plt.savefig('./snaps/sensitivty_cambridgeSIR_timeEvolution_%i_beta_%i.png'%(1,error_perc), bbox_inches='tight')
plt.savefig('./snaps/sensitivty_cambriSIR_timeEvolution_%i_beta_%i.pdf'%(1,error_perc), bbox_inches='tight')


#################################################################
######## Plots Simulation with reproductive/growth rates ########
#################################################################
do_growth = 0
if do_growth:

    # Final number of infected individuals
    IC[np.argsort(IC)[-1]]

    # Plot reproductive rates
    # matrix for linearised dynamics
    L0 = np.zeros((M, M))
    L  = np.zeros((2*M, 2*M))
    xind=[np.argsort(IC)[-1]]
    rr = np.zeros((Tf))

    for tt in range(Tf):
        Si = np.array((data['X'][tt*10,0:M])).flatten()
        for i in range(M):
            for j in range(M):
                L0[i,j]=C[i,j]*Si[i]/Ni[j]
        L[0:M, 0:M]     =    alpha*beta/gIs*L0
        L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
        L[M:2*M, 0:M]   =    ((1-alpha)*beta/gIs)*L0
        L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0

        rr[tt] = np.real(np.max(np.linalg.eigvals(L)))
        

    # compuye growth rate
    effective_Rt = rr
    growth_rates = gamma * (effective_Rt - 1)

    ####### Plots for Growth Rates #######
    fig, (ax1, ax2) = plt.subplots(1,2)

    # Plot of Reproductive rate (number)
    ax1.plot(t, effective_Rt, 'k', lw=2, label='Rt (Effective Reproductive Rate)')
    ax1.text(t[0] + 0.02, effective_Rt[0] - 0.15,r'${\cal R}_t$', fontsize=20)
    ax1.plot(t, 1*np.ones(len(t)), 'r-')
    txt1 = "Critical (Rt={per:2.2f})"
    ax1.text(t[-1]-x_axis_offset, 1 + 0.01, txt1.format(per=1), fontsize=20, color='r')
    ax1.text(t[-1]-x_axis_offset,2.5, r"${\cal R}_t \equiv \left( \frac{S (t) }{N (t) } \right) {\cal R}_0$", fontsize=20, bbox=dict(facecolor='red', alpha=0.2))


    # Estimations of End of Epidemic
    effRT_diff     = effective_Rt - 1
    ids_less_1     = np.nonzero(effRT_diff < 0)
    if len(ids_less_1)> 1:
        effRT_crossing = ids_less_1[0][0]
        ax1.plot(effRT_crossing, 1,'ro', markersize=12)
        ax1.text(effRT_crossing-10, 1-0.2,str(effRT_crossing), fontsize=20, color="r")


    ax1.set_ylabel('Rt (Effective Reproductive Rate)', fontsize=20)
    ax1.set_xlabel('Time[days]', fontsize=20)
    ax1.set_ylim(0,4)
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0^e$={R0:1.3f}, $\beta_e$={beta_:1.3f}, 1/$\gamma$={gamma_inv:1.3f})"
    fig.suptitle(txt_title.format(N=N, R0=float(r0), beta_= beta, gamma_inv = 1/gamma_inv),fontsize=20)


    # Plot of temporal growth rate
    ax2.plot(t, growth_rates, 'k', lw=2, label='rI (temporal growth rate)')
    ax2.text(t[0] + 0.02, growth_rates[0] - 0.02,r'${r}_I(t)$', fontsize=20)    
    ax2.plot(t, 0*np.ones(len(t)), 'r-')
    txt1 = r"Critical ($r_I$={per:2.2f})"
    ax2.text(t[-1]-x_axis_offset, 0 + 0.01, txt1.format(per=0), fontsize=20, color='r')
    ax2.text(t[-1]-x_axis_offset, 0.2, r"$r_I  \equiv \gamma \left[ {\cal R}_t - 1 \right]$", fontsize=20, bbox=dict(facecolor='red', alpha=0.2))
    ax2.text(t[-1]-x_axis_offset, 0.1, r"$\frac{ dI}{dt} = r_I \, I $", fontsize=20, bbox=dict(facecolor='red', alpha=0.2))
    
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
        ax2.text(rI_crossing-10, 0-0.04,str(rI_crossing), fontsize=20, color="r")


    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    plt.savefig('./snaps/cambridgeSIR_growthRates_%i.png'%ii, bbox_inches='tight')
    plt.savefig('./snaps/cambridgeSIR_growthRates_%i.pdf'%ii, bbox_inches='tight')



# fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})



# ########### Plot evolution of growth rates
# Plot reproductive rates

# # Plot reproductive rates
# Tf = 200

# fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})

# txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}"
# fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv),fontsize=20)
# plt.plot(t[::10], rr, 'o', lw=4, color='#A60628', label='suscetible', alpha=0.8,)
# plt.fill_between(t, 0, t*0+1, color="dimgrey", alpha=0.2); plt.ylabel('Basic reproductive ratio')
# plt.ylim(np.min(rr)-.1, np.max(rr)+.1)
# plt.xticks(np.arange(0, 200, 30), ('4 Mar', '3 Apr', '3 May', '2 Jun', '2 Jul', '1 Aug', '31 Aug'));
# plt.savefig('./snaps/cambridgeSIRModel_reproductiveRate.png', format='png', dpi=212)


plt.show()
