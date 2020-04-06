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
        

# initial conditions    
Is_0 = np.zeros((M));  Is_0[4:11]=4;  Is_0[1:4]=1
Ia_0 = np.zeros((M))
R_0  = np.zeros((M))
S_0  = Ni - (Ia_0 + Is_0 + R_0)


beta  = 0.01566         # infection rate 

ii = 0 

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

# the contact structure is independent of time 
def contactMatrix(t):
    return C

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



def contactMatrix(t):
    return C

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

txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0^e$={R0:1.3f}, $\beta_e$={beta_:1.3f}, 1/$\gamma$={gamma_inv:1.3f})"
fig.suptitle(txt_title.format(N=N, R0=float(r0), beta_= beta, gamma_inv = gamma_inv),fontsize=15)

# Variable evolution
plot_all = 0
if plot_all == 1:
    ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
    ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
    ax1.plot(t, R/N, 'g--',  lw=1,  label='Recovered')

    # Plot Final Epidemic Size
    ax1.plot(t, One_SinfN*np.ones(len(t)), 'm--')
    txt1 = "Final Epidemic size (no intervention): 1-S(inf)/N={per:2.2f} percentage (analytic)"
    ax1.text(t[-1]-200, One_SinfN + 0.02, txt1.format(per=One_SinfN[0]), fontsize=12, color='m')

    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as rI=cte."
    ax1.text(t[0], (total_cases/N) - 0.05, txt1.format(per=total_cases/1000000), fontsize=12, color='r')


ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')

# Plot peak points
ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)

scale = 1000000
txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} from March 4" 
txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} from March 4" 
ax1.text(peak_inf_idx+10, peak_inf/N, txt_title.format(peak_inf=peak_inf/scale, peak_days= peak_inf_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

if plot_all == 1:
    ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
    ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf/scale, peak_days= peak_inf_idx), fontsize=12, color="r", bbox=dict(facecolor='white', alpha=0.75))

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
fig.set_size_inches(27.5/2, 14.5/2, forward=True)

plt.savefig('./snaps/cambridgeSIR_timeEvolution_%i.png'%ii, bbox_inches='tight')
plt.savefig('./snaps/cambridgeSIR_timeEvolution_%i.pdf'%ii, bbox_inches='tight')




# fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})


# ########### PLot evoluation of state variables
# txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}"
# fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv),fontsize=15)

# plt.plot(t, SC*10**(-6), '-', lw=4, color='#348ABD', label='susceptible', alpha=0.8,)
# plt.fill_between(t, 0, SC*10**(-6), color="#348ABD", alpha=0.3)

# plt.plot(t, IC*10**(-6), '-', lw=4, color='#A60628', label='infected', alpha=0.8)
# plt.fill_between(t, 0, IC*10**(-6), color="#A60628", alpha=0.3)

# my_data = np.genfromtxt('data/covid-cases/india.txt', delimiter='', skip_header=6)
# day, cases = my_data[:,0], my_data[:,2]
# plt.plot(cases*10**(-6), 'ro-', lw=4, color='dimgrey', ms=16, label='data', alpha=0.5)

# plt.legend(fontsize=26); plt.grid() 
# plt.autoscale(enable=True, axis='x', tight=True)
# plt.ylabel('Individuals (millions)')
# plt.plot(t*0+t[np.argsort(IC)[-1]], -170+.4*SC*10**(-6), lw=4, color='g', alpha=0.8)
# plt.xticks(np.arange(0, 200, 30), ('4 Mar', '3 Apr', '3 May', '2 Jun', '2 Jul', '1 Aug', '31 Aug'));
# #plt.savefig('/Users/rsingh/Desktop/2b.png', format='png', dpi=212)
# plt.savefig('./snaps/cambridgeSIRModel_timeEvolution.png', format='png', dpi=212)


# # ########### Plot evolution of growth rates
# # Plot reproductive rates
# # Final number of infected individuals
# IC[np.argsort(IC)[-1]]

# # Plot reproductive rates
# Tf = 200
# # matrix for linearised dynamics
# L0 = np.zeros((M, M))
# L  = np.zeros((2*M, 2*M))
# xind=[np.argsort(IC)[-1]]
# rr = np.zeros((Tf))

# for tt in range(Tf):
#     Si = np.array((data['X'][tt*10,0:M])).flatten()
#     for i in range(M):
#         for j in range(M):
#             L0[i,j]=C[i,j]*Si[i]/Ni[j]
#     L[0:M, 0:M]     =    alpha*beta/gIs*L0
#     L[0:M, M:2*M]   = fsa*alpha*beta/gIs*L0
#     L[M:2*M, 0:M]   =    ((1-alpha)*beta/gIs)*L0
#     L[M:2*M, M:2*M] = fsa*((1-alpha)*beta/gIs)*L0

#     rr[tt] = np.real(np.max(np.linalg.eigvals(L)))


# fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams.update({'font.size': 22})

# txt_title = r"COVID-19 Cambridge SIR Model Dynamics (N={N:10.0f},$R_0$={R0:1.3f}, $\beta$={beta:1.3f}, 1/$\gamma$={gamma_inv:1.3f}"
# fig.suptitle(txt_title.format(N=N, R0=r0, beta= beta, gamma_inv = gamma_inv),fontsize=15)
# plt.plot(t[::10], rr, 'o', lw=4, color='#A60628', label='suscetible', alpha=0.8,)
# plt.fill_between(t, 0, t*0+1, color="dimgrey", alpha=0.2); plt.ylabel('Basic reproductive ratio')
# plt.ylim(np.min(rr)-.1, np.max(rr)+.1)
# plt.xticks(np.arange(0, 200, 30), ('4 Mar', '3 Apr', '3 May', '2 Jun', '2 Jul', '1 Aug', '31 Aug'));
# plt.savefig('./snaps/cambridgeSIRModel_reproductiveRate.png', format='png', dpi=212)


plt.show()
