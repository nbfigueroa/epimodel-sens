import numpy as np
from scipy.integrate  import odeint
from scipy.integrate  import solve_ivp
from scipy.optimize   import fsolve


class SEIR():
    def __init__(self, N, **kwargs):

        #Default kwargs
        self.args = {}
        
        self.args['I0']         = 1            #Initial infected population
        self.args['E0']         = 0.0001       #Initial exposed population
        self.args['R0']         = 0            #Initial immune/recovered population


        # One way of defining the model parameters
        # self.args['r0'] = 2            #reproductive rate
        # self.args['inf_period'] = 4.5  #Infection period
        # self.args['lat_period'] = 5    #Infection period

        # Another way
        self.args['beta']       = 0.33
        self.args['gamma']      = 1/7
        self.args['sigma']      = 1/5

        #Set keyword values
        for key in kwargs.keys():
            self.args[key] = kwargs[key]

        self.N = N

    @property
    def beta(self):
        return self.args['beta']

    @property
    def gamma(self):
        return self.args['gamma']

    @property
    def sigma(self):
        return self.args['sigma']

    @property
    def r0(self):
        return self.args['r0']

    def deriv(self):        
        def ddt(y,t):
            S, E, I, R = y            
            # Main state variables with exponential rates
            dSdt = -(self.beta * I * S)/self.N 
            dEdt =  (self.beta * S * I)/self.N - self.sigma*E
            dIdt =  self.sigma*E - self.gamma*I
            dRdt =  self.gamma * I        
            return dSdt, dEdt, dIdt, dRdt
        return ddt

    @staticmethod    
    def deriv_static(t,y, N, beta, gamma, sigma):
        S, I, R = y
        dSdt = -beta/N * S * I
        dEdt =  (beta*S*I)/N - sigma*E
        dIdt =  sigma*E - gamma*I
        dRdt =  gamma * I

        return dSdt, dEdt, dIdt, dRdt

    def project(self, days, solver_type='ode_int', dt = 1):

        # # A grid of time points (in simulation_time)
        t = np.arange(0, days, dt)

        #Set initial conditions
        y0 = (self.N - self.args['I0'] - self.args['R0'], self.args['E0'], self.args['I0'], self.args['R0'])

        #Integrate the ODE
        if solver_type == 'ode_int':
            ode_sol  = odeint(self.deriv(), y0, t)
            S,E,I,R  = ode_sol.T

        elif solver_type == 'solve_ivp':
            ode_sol  = solve_ivp(lambda t, y: SIR.deriv_static(t, y, self.N, self.beta, self.gamma, self.sigma), y0=y0, t_span=[0, days], t_eval=t)
            S  = ode_sol['y'][0]
            E  = ode_sol['y'][1]
            I  = ode_sol['y'][2]
            R  = ode_sol['y'][3]

        else: 
            print("INVALID SOLVER TYPE")

        return S, E, I, R
    
    def final_infection(self):
        
        def f(x):
            return np.log(x) + self.args['r0']*(1-x)
        
        answer = fsolve(f, 0.0001)
        return 1-answer


if __name__ == '__main__':
    
    seir    = SEIR(N = 10000)
    S,E,I,R = seir.project(days = 80)
