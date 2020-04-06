from sir import *
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

def rollout_Mich(N = 1375987036, days = 365, log_r0_mean = 0.44, log_r0_sd = 0.57,
                 log_gamma_mean = -2.4, log_gamma_sd = 0.73,
                 theta = [0.9999413, 3.12e-5, 2.75e-5], k = 1.5e6, time_points= [],
                 values = []):
        
        #N = 1375987036  #population of India
        #days = 365  # projection horizon starting from March 21st
        
        #sample r0
        r0 = np.random.lognormal(log_r0_mean, log_r0_sd)
        gamma = np.random.lognormal(log_gamma_mean, log_gamma_sd)
        
        kwargs['r0'] = r0
        kwargs['inf_period'] = 1/gamma
    
        #theta = np.array([1-(3.12e-5 + 2.75e-5), 3.12e-5, 2.75e-5])
        #k = 1.5e6
        
        y = np.random.dirichlet(k*theta)
        kwargs['I0'] = y[1]
        kwargs['R0'] = y[2]
        
        #assume no interventions
        #time_points = [-1,3]
        #values = [0.8, 0.6]
        
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
