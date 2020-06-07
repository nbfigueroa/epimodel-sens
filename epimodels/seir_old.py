import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats

# The SEIR model differential equations.
def seir_ivp(t, X, N, beta, gamma, sigma):
    S, E, I, R = X
    
    # Main state variables with exponential rates
    dSdt = -(beta * I * S)/N 
    dEdt =  (beta*S*I)/N - sigma*E
    dIdt  = sigma*E - gamma*I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt

def seir_ode(X, t, N, beta, gamma, sigma):
    S, E, I, R = X
    
    # Main state variables with exponential rates
    dSdt = -(beta * I * S)/N 
    dEdt =  (beta*S*I)/N - sigma*E
    dIdt  = sigma*E - gamma*I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt

# The SEIR model differential equations with mortality rates
def seir_deaths_ivp(t, X, N, beta, gamma, sigma, m):
    S, E, I, R, D = X

    dSdt  = - (beta*S*I)/N 
    dEdt  = (beta*S*I)/N - sigma*E    
    
    # Original formulation
    # dIdt  = sigma*E - gamma*I
    # dRdt  = (1-m)*gamma*I 
    # dDdt  = m*gamma*I 

    # Change of variable way --> Yields same results
    dIdt  = sigma*E - gamma*I - m*I
    dRdt  = gamma*I 
    dDdt  = m*I 

    return dSdt, dEdt, dIdt, dRdt, dDdt

def seir_deaths_ode(X, t, N, beta, gamma, sigma, m):
    return seir_deaths_ivp(t, X, N, beta, gamma, sigma, m)

def simulate_seirModel(seir_type, SEIRparams, solver_type, y0, N, simulation_time, dt):
    
    N, beta, gamma, sigma = SEIRparams 

    # A grid of time points (in simulation_time)
    t_eval = np.arange(0, simulation_time, dt)

    if seir_type == 0:
        # Standard SEIR model no deaths
        ''' Compartment structure of armed forces SEIR model
            N = S + E + I + R 
        '''
        if solver_type:
            # Integrate the SEIR equations with LSODA approach (adaptive)
            ode_sol = solve_ivp(lambda t, X: seir_ivp(t, X, N, beta, gamma, sigma), y0=y0, t_span=[0, simulation_time], t_eval=t_eval, method='LSODA', atol=1e-4, rtol=1e-6)

            t  = ode_sol['t']
            S  = ode_sol['y'][0]
            E  = ode_sol['y'][1]
            I  = ode_sol['y'][2]
            R  = ode_sol['y'][3]
        else:
            # Integrate the SEIR equations with typical approach (dopri..)
            # Using the standard ODEint function
            ode_sol = odeint(seir_ode, y0, t_eval, args=(N, beta, gamma, sigma), atol=1e-4, rtol=1e-6)
            t = t_eval
            S, E, I, R = ode_sol.T

        # Pack timeseries
        sol_ode_timeseries = np.vstack((t, S, E, I, R))

    else:
        ''' Compartment structure of armed forces SEIR model
            N = S + E + I + R + D
        '''

        # Initial conditions vector
        S0, E0, I0, R0, D0 = y0

        if solver_type:
            # Integrate the SEIR equations with LSODA approach (adaptive)
            ode_sol = solve_ivp(lambda t, X: seir_deaths_ivp(t, X, N, beta, gamma, sigma, m), y0=y0, t_span=[0, simulation_time], t_eval=t_eval, method='LSODA')

            t   = ode_sol['t']
            S   = ode_sol['y'][0]
            E   = ode_sol['y'][1]
            I   = ode_sol['y'][2]
            Re  = ode_sol['y'][3]
            D   = ode_sol['y'][4]

        else:     
            # Integrate the SEIR equations with typical approach (dopri..)
            # Using the standard ODEint function
            ode_sol = odeint(seir_deaths_ode, y0, t_eval, args=(N, beta, gamma, sigma. m))
            t = t_eval
            S, E, I, Re, D = ode_sol.T
        
        # Pack timeseries            
        sol_ode_timeseries = np.vstack((t, S, E, I, Re, D))    

    return  sol_ode_timeseries  
