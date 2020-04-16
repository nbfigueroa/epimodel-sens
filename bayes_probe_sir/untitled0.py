# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:46:50 2020

@author: AJShah
"""

from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import numpy as np

def solve_prevalence(N_tests, threshold = 0.01):
    #Solve for the prevalence where the probability of non-detection falls
    #below threshold
    
    cdf_val = 1-threshold
    
    prevalence = poisson.ppf(cdf_val, 1/p)
    return prevalence
    
    

if __name__ == '__main__':
    
    N = 50e-6
    prevalence = solve_prevalence(20000)
    
    print('The prevalence must be more than {prevalence} to have less than 1% chance of missing cases')
    
    '''
    N = 50e6
    
    prevalence = np.logspace(np.log10(1e-9), np.log10(1), 10000)
    N_tests = np.logspace(1, 5, 5)
    P_zero = {}
    
    for n in N_tests:
        P_zero[n] = [1-poisson.cdf(n,1/p) for p in prevalence]
    
    with sns.plotting_context('poster'):
        plt.figure(figsize = [15,10])
        for n in N_tests:
            plt.semilogx(prevalence*N, P_zero[n], label = f'{int(n)} Tests')
        plt.legend()
        plt.xlabel('Number of cases in India')
        plt.ylabel('Probability of zero positive results')
        plt.title('Case Detection with Random Testing')
        
    prevalence = np.logspace(np.log10(1e-9), np.log10(1), 1000)
    N_tests = np.logspace(0,5,100)
    
    vals = list(product(prevalence, N_tests))
    p_zero = np.zeros((len(prevalence), len(N_tests)))
    
    for (i,p) in enumerate(prevalence):
        for (j,n) in enumerate(N_tests):
            p_zero[i,j] = 1 - poisson.cdf(n,1/p)
    
    plt.figure(figsize = [15,10])
    plt.contourf(N_tests, prevalence, p_zero)
    plt.xscale('log')
    plt.yscale('log')
    '''