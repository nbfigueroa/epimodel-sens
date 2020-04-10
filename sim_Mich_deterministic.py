from epimodels.sir import *
from tqdm import tqdm
from epimodels.esir import * 



if __name__ == '__main__':
    
    
    #Using the params for the Mumbai model

    proportional = False
    
    N = 1375987036  #population of India
    days = 365  # projection horizon starting from March 21st
    
   
    #Control params
    # No Intervention pi = 1
    # Travel ban pi = 0.8 March 13
    # Social Distancing guidelines = 0.6 March 17
    # Total lockodown pi = 0.2 March 24
    
    
    #Disease propogation
    kwargs = {}
    kwargs['r0'] = 2.28 # the mean parameter
    kwargs['inf_period'] = 7 #mean infection period, 1/gamma, 7 days
    
    #Initial population params
    kwargs['R0'] = 23 #Recovered patients as of March 21st
    Q0 = 249 # Detected cases and quarantined as of March 21st. Assumed to be 1% of total infections
    q = 0.01 #1% of true cases quarantined
    kwargs['I0'] = (1.01/0.01)*Q0 # Assumption that only 1% of cases have been detected and quarantined
    
    time_points = [3] #Nationwide lockdown imposed here
    values = [0.2]
    
    #Initialize the model
    model1 = eSIR(N,time_points, values, **kwargs)
    S1, I1, R1 = model1.project(days)
    
    create_plots(model1, S1, I1, R1, proportional = False)
    
    
        
    #Recreating the UMich report numbers, need to get the posterior mean 
    #estimate of R0 from data per the report
    
    kwargs['r0'] = 1.78  #Using the stochastic estimator to estimate params
    kwargs['inf_period'] = 1/0.119 #from estimated posterior
    
    #Start date is March 16
    time_points = []
    values = []
    
    #Population params
    kwargs['I0'] = 127#Assuming that all cases are detected
    kwargs['R0'] = 14 #
    
    model2 = eSIR(N, time_points, values, **kwargs)
    S2,I2,R2 = model2.project(60)
    
    create_plots(model2, S2, I2, R2, proportional = False, startdate = 'March 16')


    
    plt.figure(figsize = [10,8])
    plt.title('Michigan: Cumulative Cases', fontsize = 20)
    plt.xlabel('Days after March 16', fontsize = 18)
    plt.ylabel('Proportion of population', fontsize = 18)
    plt.semilogy(N - S_mean[0:60], label = 'Infected')
    plt.fill_between(np.arange(len(S_mean[0:60])), N - S_up[0:60], N-S_down[0:60], alpha = 0.5)
    
    
    plt.figure(figsize = [10,8])
    plt.title('Active Infections (Deterministic SIR with Michigan model estimates)', fontsize=20)
    plt.xlabel('Days after March 16', fontsize = 18)
    plt.ylabel('Proportion of population', fontsize = 18)
    plt.plot(I_mean/N, label = 'Infected')
    plt.fill_between(np.arange(len(I_mean)), I_up/N, I_down/N, alpha = 0.5)
    
    plt.show()