# -*- coding: utf-8 -*-
from sir import *
from scipy.stats import *


class stochastic_sir(sir):
    
    def __init__(N, beta_dist:rv_continuous = None, 
                 gamma_dist:rv_continuous = None,
                 sigma_distribution:rv_continuous = None):
        #Enter default distributions
        if beta_dist == None:
            self.beta_dist = uniform()
        