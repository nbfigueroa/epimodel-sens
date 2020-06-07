import numpy as np
from   scipy.integrate import odeint
from   scipy.integrate import solve_ivp
from   scipy.optimize import fsolve
from   scipy import stats

# The SEIR model differential equations with mortality rates and quarantine
def seiqr_mumbai_ivp(t, X, N, beta, gamma, sigma, m, q, tau_q):
    S, E, I, Q, R, D = X

    # Original State equations for SEIR
    dSdt  = - (beta*S*I)/N 
    dEdt  = (beta*S*I)/N - sigma*E    

    # Incorporating Quarantine components
    dIdt  = sigma*E - gamma*I - q*I - m*I
    dQdt  = q*I - tau_q*Q - m*Q
    dRdt  = gamma*I + tau_q*Q
    dDdt  = m*I + m*Q 

    return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt

def seiqr_mumbai_ode(t, X, N, beta, gamma, sigma, m, q, tau_q):
    return seir_deaths_ode(X, t, N, beta, gamma, sigma, m)


def simulate_seiqrModel(SEIQRparams, solver_type, y0, N, simulation_time, dt):
    
    N, beta, gamma, sigma, m, q, tau_q = SEIQRparams

    # A grid of time points (in simulation_time)
    t_eval = np.arange(0, simulation_time, dt)

    ''' Compartment structure of armed forces SEIR model
        N = S + E + I + Q + R + D
    '''

    # Initial conditions vector
    S0, E0, I0, Q0, R0, D0 = y0

    if solver_type:
        # Integrate the SEIQR equations over the time grid, with bet
        ode_sol = solve_ivp(lambda t, X: seiqr_mumbai_ivp(t, X, N, beta, gamma, sigma, m, q, tau_q), y0=y0, t_span=[0, simulation_time], t_eval=t_eval, method='LSODA')

        t   = ode_sol['t']
        S   = ode_sol['y'][0]
        E   = ode_sol['y'][1]
        I   = ode_sol['y'][2]
        Q   = ode_sol['y'][3]
        Re  = ode_sol['y'][4]
        D   = ode_sol['y'][5]


    else:     
        # Integrate the SEIR equations with typical approach (dopri..)
        # Using the standard ODEint function
        ode_sol = odeint(seiqr_mumbai_ode, y0, t_eval, args=(N, beta, gamma, sigma. m))
        t = t_eval
        S, E, I, Q, Re, D = ode_sol.T
    
    # Pack timeseries            
    sol_ode_timeseries = np.vstack((t, S, E, I, Q, Re, D))    

    return  sol_ode_timeseries  
