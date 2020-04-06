from bayes_probe_sir.sir import *
from tqdm import tqdm

class eSIR(SIR):
    
    def __init__(self, N, time_points, values, **kwargs):
        
        super().__init__(N,**kwargs)
        assert len(time_points) == len(values)
        self.time_points = time_points
        self.values = values
    
    def determine_scaling(self, t):
        
        if len(self.time_points) > 0:
            idx = np.argmin(np.abs(np.array(self.time_points) - t))
            closest_time = self.time_points[idx]
            
            #value = self.values[idx]
            
            if closest_time > t:
                
                if idx==0:
                    return 1
                else:
                    return self.values[idx-1]
            else:
                
                return self.values[idx]
        else:
            return 1
    
    
    def deriv(self):
        def ddt(y,t):
            S, I, R = y
            pi = self. determine_scaling(t)
            dSdt = -(self.beta*pi)/self.N * S * I
            dIdt = (self.beta*pi)/self.N * S * I  - self.gamma * I
            dRdt = self.gamma * I
            return dSdt, dIdt, dRdt
        return ddt

def create_plots(model, S,I,R, proportional=False, startdate = 'March 21'):
        #Plot S,I,R populations on the same graph
        if proportional:
            plt.plot(I/model.N, label = 'Infected')
            plt.plot(R/model.N, label = 'Resolved')
        else:
            plt.figure(figsize = [10,8])
            
            #plt.plot(S1, label = 'S')
            plt.plot(I, label = 'Infected')
            plt.plot(R, label = 'Resolved')
            #plt.plot([0, days], [N,N], 'k')
        plt.legend(prop = {'size':14})
        plt.xlabel(f'Days after {startdate}', fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.title(f'Cases after {startdate}', fontsize = 20)
        plt.ylabel('Number', fontsize = 18)
        plt.yticks(fontsize = 14)
        
        #Compute the final number of infetions
        print('The number of final infected cases: ', model.N - S[-1])
        
        #Plot the r(t) param, should cross 0 when the epidemic peaks
        plt.figure(figsize = [10,8])
        r_t = []
        for (i,s) in enumerate(S):
            scaling = model.determine_scaling(i)
            val = scaling*model.args['r0']*s/model.N
            r_t.append(val)
        
        plt.plot(r_t)
        plt.plot([0, len(r_t)], [1,1], 'k')
        plt.title('Effective Reproductive Rate', fontsize = 20)
        plt.xlabel(f'Days after {startdate}')
        
        #Report the maximum number of infections
        day = np.argmax(I)
        print('The maximum number of active infections at any time is: ', np.max(I))
        
        print(f'Maximum case load is expected after {day} days')

def rollout_Mich():
        
        N = 1375987036  #population of India
        days = 365  # projection horizon starting from March 21st
        
        #sample r0
        r0 = np.random.lognormal(0.31,0.73)
        gamma = np.random.lognormal(-2.4, 0.73)
        
        kwargs['r0'] = r0
        kwargs['inf_period'] = 1/gamma
    
        theta = np.array([1-(3.12e-5 + 2.75e-5), 3.12e-5, 2.75e-5])
        k = 1.5e6
        
        y = np.random.dirichlet(k*theta)
        kwargs['I0'] = y[1]*N
        kwargs['R0'] = y[2]*N
        
        #assume no interventions
        time_points = []
        values = []
        
        model = eSIR(N, time_points, values, **kwargs)
        
        S,I,R = model.project(days)
        
        return r0, gamma, model, S, I, R

def MC_Mich(n_rollouts):
    
    r0s = []
    gammas = []
    models = []
    S = []
    I = []
    R = []
    for i in tqdm(range(n_rollouts)):
        r0, gamma, model, S_new, I_new, R_new = rollout_Mich()
        r0s.append(r0)
        gammas.append(gamma)
        models.append(model)
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
    S = np.array(S)
    I = np.array(I)
    R = np.array(R)
    return r0s, gammas, models, S, I, R



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
    kwargs['I0'] = 3.12e-5*N #Assuming that all cases are detected
    kwargs['R0'] = 2.75e-5*N #
    
    model2 = eSIR(N, time_points, values, **kwargs)
    S2,I2,R2 = model2.project(60)
    
    create_plots(model2, S2, I2, R2, proportional = False, startdate = 'March 16')
    60
    
    r0s, gammas, models, S, I, R = MC_Mich(10000)
    S_mean = np.mean(S, axis=0)
    I_mean = np.mean(I, axis = 0)
    R_mean = np.mean(R, axis = 0)
    
    S_up = np.quantile(S, 0.975, axis = 0)
    S_down = np.quantile(S, 0.025, axis = 0)
    
    I_up = np.quantile(I, 0.975, axis = 0)
    I_down = np.quantile(I, 0.025, axis = 0)
    
    R_up = np.quantile(R, 0.975, axis=0)
    R_down = np.quantile(R, 0.025, axis = 0)
    
    plt.figure(figsize = [10,8])
    plt.title('Michigan: Cumulative Cases', fontsize = 20)
    plt.xlabel('Days after March 16', fontsize = 18)
    plt.ylabel('Proportion of population', fontsize = 18)
    plt.plot(N - S_mean, label = 'Infected')
    plt.fill_between(np.arange(len(S_mean)), N - S_up, N - S_down, alpha = 0.5)
    
    
    plt.figure(figsize = [10,8])
    plt.title('Michigan: Active Infections')
    plt.xlabel('Days after March 16', fontsize = 18)
    plt.ylabel('Proportion of population', fontsize = 18)
    plt.plot(I_mean/N, label = 'Infected')
    plt.fill_between(np.arange(len(I_mean)), I_up/N, I_down/N, alpha = 0.5)
    