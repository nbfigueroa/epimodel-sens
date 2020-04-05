import numpy as np
import pyross
import matplotlib.pyplot as plt
from scipy.io import loadmat

M  = 2                  # the population has two age groups
N  =  1000000           # and this is the total population

beta  = 0.0131          # infection rate
gIa   = 1./7            # recovery rate of asymptomatic infectives 
gIs   = 1./7            # recovery rate of asymptomatic infectives 
alpha = 0               # fraction of asymptomatic infectives 
fsa   = 1               # the self-isolation parameter   


Ni = np.zeros((M))      # population in each group
fi = np.zeros((M))      # fraction of population in age age group

# set the age structure
fi = np.array((0.25, 0.75)) 
for i in range(M):
    Ni[i] = fi[i]*N
    
# set the contact structure
C = np.array(([18., 9.], [3., 12.]))
    
    
Ia_0 = np.array((1,1))  # each age group has asymptomatic infectives
Is_0 = np.array((1,1))  # and also symptomatic infectives 
R_0  = np.array((0,0))  # there are no recovered individuals initially
S_0  = Ni - (Ia_0 + Is_0 + R_0)


# matrix for linearised dynamics
L = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        L[i,j]=C[i,j]*Ni[i]/Ni[j]

L = (alpha*beta/gIs)*L

# the basic reproductive ratio
r0 = np.max(np.linalg.eigvals(L))

print("The basic reproductive ratio for these parameters is", r0)


# duration of simulation and data file
Tf=200; Nf=2000; filename='this.mat'

# the contact structure is independent of time 
def contactMatrix(t):
    return C

# instantiate model
parameters = {'alpha':alpha, 'beta':beta, 'gIa':gIa, 'gIs':gIs,'fsa':fsa}
model = pyross.models.SIR(parameters, M, Ni)


# simulate model
model.simulate(S_0, Ia_0, Is_0, contactMatrix, Tf, Nf, filename)


data=loadmat(filename)
IK = data['X'][:,2*M].flatten()
IA = data['X'][:,2*M+1].flatten()
t = data['t'][0]


# Plots infected segmented by kids and adults
fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})

plt.fill_between(t, 0, IK/Ni[0], color="#348ABD", alpha=0.3)
plt.plot(t, IK/Ni[0], '-', color="#348ABD", label='$Children$', lw=4)

plt.fill_between(t, 0, IA/Ni[1], color='#A60628', alpha=0.3)
plt.plot(t, IA/Ni[1], '-', color='#A60628', label='$Adults$', lw=4)

plt.legend(fontsize=26); plt.grid() 
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()

