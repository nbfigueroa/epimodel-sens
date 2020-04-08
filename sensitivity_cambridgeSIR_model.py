import pyross
import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats
from   scipy.io import loadmat
from   matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter

from   epimodels.utils import *

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


####################################
####  Fixed SIR Model Paramaters  ##
####################################

###### Initial Conditions for Simulation   ######
Is_0 = np.zeros((M));  Is_0[4:11]=4;  Is_0[1:4]=1
Ia_0 = np.zeros((M))
R_0  = np.zeros((M))
S_0  = Ni - (Ia_0 + Is_0 + R_0)


###### Fixed values for simulation  ######
gIa   = 1./7            # recovery rate of asymptomatic infectives 
gIs   = 1./7            # recovery rate of symptomatic infectives 
alpha = 0.              # fraction of asymptomatic infectives 
fsa   = 1               # the self-isolation parameter   
gamma =  1/gIs       
gamma_inv = gIs
C=CH+CW+CS+CO           # matrix of total contacts

###### Specify scenarios and beta values to test ################
scenario = 2

# infection rate 
beta_cambridge  = 0.01566         

# Variables for beta samples
error_perc = 10
beta_error = error_perc/100
beta_mean  = beta_cambridge
beta_sigma = beta_cambridge*(beta_error)
beta_plus  = beta_mean + beta_sigma
beta_minus = beta_mean - beta_sigma

beta_samples = [beta_mean, beta_plus, beta_minus]

###### Other simulation parameyters ######
# simulation time
# Tf=365
Tf= 712

# A grid of time points (in simulation_time)
t = np.arange(0, Tf, 1)


# Plotting and storing parameters
x_axis_offset       = 250
y_axis_offset       = 0.0000000003
store_plots         = 1 
plot_all            = 1
plot_peaks          = 1
show_S              = 0
show_R              = 0
plot_superimposed   = 1
store_values        = 1
show_analytic_limit = 0
do_growth           = 0


######## Record predictions ########
S_samples       = np.empty([3, Tf])
I_samples       = np.empty([3, Tf])
R_samples       = np.empty([3, Tf])
file_extensions  = ["./results/Cambridge_Scenario{scenario:d}".format(scenario=scenario), 
                    "./results/Cambridge_Scenario{scenario:d}_beta{error:d}error_plus".format(scenario=scenario, error=error_perc),
                    "./results/Cambridge_Scenario{scenario:d}_beta{error:d}error_minus".format(scenario=scenario, error=error_perc),]
header = ['beta', 'R0', 'I_42', 'T_42', 'I_70', 'T_70', 'I_90', 'T_90', 'I_120', 'T_120', 't_low', 't_c', 'I_peak', 'T_inf']
time_checkpoints = [42, 70, 90, 120] 

if store_values:
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(file_extensions[0]+".xlsx")
    worksheet = workbook.add_worksheet()
    for jj in range(len(header)):
        worksheet.write(0, jj,  header[jj] )    


######## Start simulations ########
for ii in range(3):
    #####  beta value to evaluate #####  
    beta = beta_samples[ii]

    ##### Compute R0 (not really needed to run the simulation but will differ for beta values) #####
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

    print('*****   Model Parameters    *****')
    print("beta_bar", beta, "RO^eff", r0, "1/gamma", gIs)

    if show_analytic_limit:
        # Estimated Final epidemic size (analytic) not-dependent on simulation
        init_guess   = 0.0001
        r0_test      = float(r0)
        SinfN  = fsolve(epi_size, init_guess)
        One_SinfN = 1 - SinfN
        print('*****   Final Analytic Epidemic Size    *****')
        print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

    if store_values:
        # Store beta and r
        worksheet.write(ii+1, 0,  beta )
        worksheet.write(ii+1, 1,  float(r0) )

    ###### Initial conditions ######  
    Is_0 = np.zeros((M));  Is_0[4:11]=4;  Is_0[1:4]=1
    Ia_0 = np.zeros((M))
    R_0  = np.zeros((M))
    S_0  = Ni - (Ia_0 + Is_0 + R_0)

    ##############################
    ####  SIR Model Simulation  ##
    ##############################
    # intantiate model
    parameters = {'alpha':alpha,'beta':beta, 'gIa':gIa,'gIs':gIs,'fsa':fsa}
    model = pyross.models.SIR(parameters, M, Ni)

    # No intervention
    if scenario == 0:
        def contactMatrix(t):
            return C
    # Lockdown for 21 days            
    if scenario == 1:
        def contactMatrix(t):
            if t<21:
                xx = C
            elif 21<=t<42:
                xx = CH
            else:
                xx = C
            return xx
    # Lockdown for 49 days (no assumption of quarantine afterwards)
    if scenario == 2:
        def contactMatrix(t):
            if t<21:
                xx = C
            elif 21<=t<70:
                xx = CH
            else:
                xx = C
            return xx

    # Lockdown for 79 days (no assumption of quarantine afterwards)
    if scenario == 3:
        plot_all = 1
        def contactMatrix(t):
            if t<21:
                xx = C
            elif 21<=t<100:
                xx = CH
            else:
                xx = C
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

    # Storing run in matri for post-processing
    S_samples[ii,:] = S
    I_samples[ii,:] = I
    R_samples[ii,:] = R

    print('*********   Results    *********')    
    if store_values:
        # Store Intervals
        for jj in range(len(time_checkpoints)):
            worksheet.write(ii+1, (jj+1)*2,  I[time_checkpoints[jj]])
            worksheet.write(ii+1, (jj+1)*2 + 1,  T[time_checkpoints[jj]])
            print('I(',time_checkpoints[jj],')=',I[time_checkpoints[jj]], 'T(',time_checkpoints[jj],')=',T[time_checkpoints[jj]])

    Ids_less_10  = np.nonzero(I < 11)
    I_less_10 =  Ids_less_10[0][0]
    print('I(t_low)=',I_less_10)

    peak_inf_idx =  np.argmax(I)
    peak_inf     = I[peak_inf_idx]
    print('Peak Instant. Infected = ', peak_inf,'by day=', peak_inf_idx)

    peak_total_inf  = T[peak_inf_idx]
    print('Total Cases @ Peak = ', peak_total_inf,'by day=', peak_inf_idx)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)
        
    if store_values:
        # Store results
        worksheet.write(ii+1, 10,  I_less_10)
        worksheet.write(ii+1, 11,  peak_inf_idx)
        worksheet.write(ii+1, 12,  peak_inf)
        worksheet.write(ii+1, 13,  total_cases)

    #####################################################################
    ######## Plots Simulation with point estimates of parameters ########
    #####################################################################
    txt_title    = r"COVID-19 Cambridge SIR Model Dynamics [Scenario {scenario:d}] ($R_0^e$={R0:1.3f}, $\beta_e$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
    SIRparams    = scenario, float(r0), beta, gamma_inv, N
    SIRvariables = S, I, R, T, t
    plot_all_ii   = 1
    Plotoptions  = plot_all_ii, show_S, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset
    plotSIR_evolution(txt_title, SIRparams, SIRvariables, Plotoptions, store_plots, file_extensions[ii])


if store_values:
    workbook.close()

#####################################################################
######## Plots Simulation with point estimates of parameters ########
#####################################################################
if plot_superimposed:
    Plotoptions  = plot_all, show_S, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, beta_error
    plotSIR_evolutionErrors(txt_title, SIRparams, S_samples, I_samples, R_samples, Plotoptions, store_plots, file_extensions[0])

############################################################################
######## TO CHECK : Plots Simulation with reproductive/growth rates ########
############################################################################
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

    if store_plots:
        plt.savefig('./snaps/cambridgeSIR_growthRates_%i.png'%ii, bbox_inches='tight')
        plt.savefig('./snaps/cambridgeSIR_growthRates_%i.pdf'%ii, bbox_inches='tight')


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



# if scenario == 3:
#     def contactMatrix(t):
#         if t<21:
#             xx = C
#         elif 21<=t<70:
#             xx = CH
#         else:
#             xx = CH + CW
#         return xx
# if scenario == 4:
#     def contactMatrix(t):
#         if t<21:
#             xx = C
#         elif 21<=t<70:
#             xx = CH
#         else:
#             xx = CH + CW + CS
#         return xx

# if scenario == 5:
#     def contactMatrix(t):
#         if t<21:
#             xx = C
#         elif 21<=t<120:
#             xx = CH
#         else:
#             xx = CH + CW + CS
#         return xx

# if scenario == 6:
#     def contactMatrix(t):
#         if t<21:
#             xx = C
#         elif 21<=t<42:
#             xx = CH
#         elif 42<=t<47:
#             xx = C
#         elif 47<=t<75:
#             xx = CH
#         else:
#             xx = C
#         return xx


