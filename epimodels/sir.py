import numpy as np
from scipy.integrate  import odeint
from scipy.integrate  import solve_ivp
from scipy.optimize   import fsolve


class SIR():

    def __init__(self, N, **kwargs):

        #Default kwargs
        self.args = {}
        self.args['r0'] = 2 #reproductive rate
        self.args['inf_period'] = 4.5 #Infection period
        self.args['I0'] = 1 #Initial infected population
        self.args['R0'] = 0 #Initial immune/recovered population

        #Set keyword values
        for key in kwargs.keys():
            self.args[key] = kwargs[key]

        self.N = N

    @property
    def beta(self):
        return self.args['r0']/self.args['inf_period']

    @property
    def gamma(self):
        return 1/self.args['inf_period']

    @property
    def r0(self):
        return self.args['r0']

    def deriv(self):
        def ddt(y,t):
            S, I, R = y
            dSdt = -self.beta/self.N * S * I
            dIdt = self.beta/self.N * S * I  - self.gamma * I
            dRdt = self.gamma * I
            return dSdt, dIdt, dRdt
        return ddt

    @staticmethod    
    def deriv_static(t,y, N, beta, gamma):
        S, I, R = y
        dSdt = -beta/N * S * I
        dIdt = beta/N * S * I  - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def project(self, days, solver_type='ode_int', dt = 1):

        # # A grid of time points (in simulation_time)
        t = np.arange(0, days, dt)

        #Set initial conditions
        y0 = (self.N - self.args['I0'] - self.args['R0'], self.args['I0'], self.args['R0'])

        #Integrate the ODE
        if solver_type == 'ode_int':
            ode_sol  = odeint(self.deriv(), y0, t)
            S,I,R    = ode_sol.T

        elif solver_type == 'solve_ivp':
            ode_sol  = solve_ivp(lambda t, y: SIR.deriv_static(t, y, self.N, self.beta, self.gamma), y0=y0, t_span=[0, days], t_eval=t)
            S  = ode_sol['y'][0]
            I  = ode_sol['y'][1]
            R  = ode_sol['y'][2]

        else: 
            print("INVALID SOLVER TYPE")

        return S, I, R
    
    def final_infection(self):
        
        def f(x):
            return np.log(x) + self.args['r0']*(1-x)
        
        answer = fsolve(f, 0.0001)
        return 1-answer


if __name__ == '__main__':
    
    sir   = SIR(N = 10000)
    S,I,R = sir.project(days = 80)
