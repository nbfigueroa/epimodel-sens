import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class SIR():

    def __init__(self, N, **kwargs):

        #Default kwargs
        self.args = {}
        self.args['r0'] = 2.65 #reproductive rate
        self.args['inf_period'] = 4.5 #Infection period
        self.args['I0'] = 1 #Initial infected population
        self.args['R0'] = 0 #Initial immune/recovered population

        #Set keyword values
        for key in kwargs.keys():
            self.args[key] = kwargs[key]

        self.N = N
        #self.days = days

    @property
    def beta(self):
        return self.args['r0']/self.args['inf_period']

    @property
    def gamma(self):
        return 1/self.args['inf_period']


    def deriv(self):
        def ddt(y,t):
            S, I, R = y
            dSdt = -self.beta/self.N * S * I
            dIdt = self.beta/self.N * S * I  - self.gamma * I
            dRdt = self.gamma * I
            return dSdt, dIdt, dRdt
        return ddt

    def project(self, days):
        #Set initial conditions
        y0 = (self.N - self.args['I0'] - self.args['R0'], self.args['I0'],self.args['R0'])

        t = np.arange(days)
        t = np.append(t,days)

        #Integrate the ODE
        ode_sol = odeint(self.deriv(), y0, t)
        S,I,R = ode_sol.T
        return S,I,R
    
    def final_infection(self):
        
        def f(x):
            return np.log(x) + self.args['r0']*(1-x)
        
        answer = fsolve(f, 0.0001)
        return 1-answer


if __name__ == '__main__':
    
    sir = SIR(10000, 80)
    S,I,R = sir.project()
