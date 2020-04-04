from seirsplus.models import *


# Values used in the armed forces paper
# N            = 1486947036
N            = 1375947036
days         = 356
gamma_inv    = 7  
kappa_inv    = 5
m            = 0.0043
r0           = 2.28      

# Initial values from March 21st "armed forces predictions"
R0           = 23
D0           = 5 
T0           = 334  
Q0           = 249 #Q0 is 1% of infectious I0
I0           = Q0*100

# Initial values from March 12th "first death!"
# R0           = 4 
# D0           = 1 
# T0           = 82  

# Calculating number of active infected and exposed
# I0           = T0 - R0 - D0

# From first infected (January 30)
# Q0 = 3
# I0 = 1
# R0 = 0 
# D0 = 0

# Derived Model parameters and Control variable 
# control: percentage of infected being quarantined
beta       = r0 / gamma_inv
kappa      = 1.0 / kappa_inv
gamma      = 1.0 / gamma_inv


pecrt = 0.19 * N
print(pecrt)

# Model with no interventions
# model = SEIRSModel(beta=0.147, sigma=1/5.2, gamma=1/12.39, mu_I=0.0004, initI=10000, initN=1000000) 
model = SEIRSModel(beta=beta, sigma=kappa, gamma=gamma, mu_I=m, initI=I0, initR=R0, initF=D0, initN=N) 
model.run(T=356, dt=1, verbose=True)
model.figure_infections()

