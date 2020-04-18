# To read data from xls sheet
from pandas import DataFrame, read_csv
import pandas as pd 
from epimodels.utils import *

def loadSimulationParams(sim_num, scenario, plot_data = 1):
    ''' Setups the model parameters and initial conditions for different simulation options
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Primer case study
    '''
    # India Simulation starting on March 4th
    if sim_num == 1:
        N               = 1353344709
        days            = round(30.5*10)
        gamma_inv       = 7  
        r0              = 2.28      
        beta            = r0 / gamma_inv
        country         = 'India'

    # Mexico Simulation starting on March 4th
    if sim_num == 2:        
        N               = 128932753
        days            = round(30.5*10)
        gamma_inv       = 7  
        r0              = 2.28      
        beta            = r0 / gamma_inv
        country         = 'Mexico'

    # US Simulation starting on March 4th
    if sim_num == 3:
        N               = 331002651
        days            = round(30.5*10)
        gamma_inv       = 7  
        r0              = 2.28      
        beta            = r0 / gamma_inv
        country         = 'US'

    # Load number of infected cases from CSSE @JHU Dataset for country case studies   
    if sim_num < 4:    
        data            = loadCSSEData(country, plot_data)    
        sim_init        = 42   # Days between Jan 22 and March 4
        Infected        = data[0,sim_init:] - data[1,sim_init:] - data[2,sim_init:]     
        I0              = data[0,sim_init] - data[1,sim_init] - data[2,sim_init]
        R0              = data[1,sim_init]  
        x_tick_names    = ('4 Mar', '4 Apr', '4 May', '4 Jun', '4 Jul', '4 Aug', '4 Sept', '4 Oct', '4 Nov', '4 Dec', '4 Jan')        
        x_tick_step     = 30

    # Yucatan Simulation Starting March 13th
    if sim_num == 4:    
        N               = 2000000 
        days            = round(30.5*7)
        gamma_inv       = 7                
        r0              = 2.28    
        beta            = r0 / gamma_inv
        I0, R0          = 1, 0    
        country         = 'Yucatan'
        x_tick_names    = ('13 Mar', '13 Apr', '13 May', '13 Jun', '13 Jul', '13 Aug', '13 Sept', '13 Oct')
        x_tick_step     = 30
        # Load number of infected cases
        Infected, Infected_est = loadYucatanData()    

    
    if sim_num  < 5:        
        figure_title    = r"COVID-19 SIR Model Dynamics [Scenario {scenario:d}] -- {country} -- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
        figure_title    = figure_title.format(scenario=scenario, country=country, R0=float(r0), beta= beta, gamma = gamma_inv)
    else:    
        # Test Population for primer paper
        N               = pow(10,5)
        days            = 200
        gamma_inv       = 7.0  
        r0              = 2.31  
        I0, R0          = 1, 0    
        beta            = r0 / gamma_inv
        country         = 'Primer'
        x_tick_names    = []
        x_tick_step     = []
        Infected        = np.array([])
        figure_title    = r"SIR Model Dynamics -- {country} -- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
        figure_title    = figure_title.format(country=country, R0=float(r0), beta= beta, gamma = gamma_inv)


    file_extension  = "./results/{country_dir}/VanillaSIR_Scenario{scenario:d}_{country}_{days:d}".format(country_dir= country.lower(), scenario=scenario, country= country, days = days)    
    store_plots         = 1
    do_growth           = 1

    sim_kwargs = {}
    sim_kwargs['N']                   = N
    sim_kwargs['days']                = days
    sim_kwargs['beta']                = beta
    sim_kwargs['r0']                  = r0
    sim_kwargs['gamma_inv']           = gamma_inv
    sim_kwargs['I0']                  = I0
    sim_kwargs['R0']                  = R0
    sim_kwargs['file_extension']      = file_extension
    sim_kwargs['figure_title']        = figure_title
    if len(x_tick_names) > 0:
        sim_kwargs['x_tick_names']    = x_tick_names
        sim_kwargs['x_tick_step']     = x_tick_step
    sim_kwargs['Infected']            = Infected
    sim_kwargs['store_plots']         = store_plots
    sim_kwargs['do_growth']           = do_growth

    return sim_kwargs


def getSIRTestingParams(test_num, prob_type,**sim_kwargs):
    '''
        Generate the parameters for different testing 
        and probability distribution variants
    '''

    #####################################################################
    ########### Method 1: Drawing from Uniform Distributions ############
    #####################################################################
    if prob_type  == 'uniform':
        beta_min      = 0.24
        beta_max      = 0.42
        gamma_inv_min = 4   
        gamma_inv_max = 10           

        ########### Test 1: Sample beta, fix gamma ############
        if test_num == 1:
            # this is an indicator that should be max value
            gamma_inv_min = sim_kwargs['gamma_inv']    
            gamma_inv_max = sim_kwargs['gamma_inv'] 
            _ext          = "_errorsSampleBeta_Uniform"
            text_error  =  r"$\beta \sim \mathcal{U}(0.24,0.42)$"

        ############ Test 2: fix beta, Sample gamma ############
        if test_num == 2:
            beta_min      = sim_kwargs['beta']
            beta_max      = sim_kwargs['beta']
            _ext          = "_errorsSampleGamma_Uniform"
            text_error  =  r"$\gamma^{-1} \sim \mathcal{U}(4,10)$"

        ############ Test 3: sample beta, sample gamma ############
        if test_num == 3:
            _ext          = "_errorsSampleBetaGamma_Uniform"            
            text_error  =  r"$\beta \sim \mathcal{U}(0.24,0.42)$"  + "\n" + r"$\gamma^{-1} \sim \mathcal{U}(4,10)$"
        
        prob_params = (prob_type, beta_min, beta_max, gamma_inv_min, gamma_inv_max) 

    ######################################################################
    ########### Method 2: Drawing from Gaussian Distributions ############
    ######################################################################
    if prob_type  == 'gaussian':           
        ## For variation on these parameters
        beta                  = sim_kwargs['beta']
        gamma_inv             = sim_kwargs['gamma_inv'] 
        beta_mean             = beta        
        gamma_inv_mean        = gamma_inv    

        # TODO: Compute the std's with equations from Dave's report
        beta_std              = 0.09
        gamma_inv_std         = 1.73 
        
        ########### Test 1: Sample beta, fix gamma ############
        if test_num == 1:
            gamma_inv_std = 0.00
            _ext          = "_errorsSampleBeta_Gaussian"

        ############ Test 2: fix beta, Sample gamma ############
        if test_num == 2:
            beta_std      = 0.00
            _ext          = "_errorsSampleGamma_Gaussian"

        ############ Test 3: sample beta, sample gamma ############
        if test_num == 3:
            _ext          =  "_errorsSampleBetaGamma_Gaussian"            
        
        text_error  =  r"$\beta \sim \mathcal{N}(\bar{\beta},%2.2f)$"%beta_std + "\n" +  r"$\gamma^{-1} \sim \mathcal{N}(\gamma^{-1},%2.2f)$"%gamma_inv_std
        prob_params = (prob_type, beta_mean, beta_std, gamma_inv_mean, gamma_inv_std) 
    

    ######################################################################
    ########### Method 2: Drawing from Gaussian Distributions ############
    ######################################################################
    if prob_type  == 'gamma':           
        beta_loc, beta_shape, beta_scale                = 0.24, 3, 0.045
        gamma_inv_loc, gamma_inv_shape, gamma_inv_scale = 4,    4, 0.865
        
        ########### Test 1: Sample beta, fix gamma ############
        if test_num == 1:
            gamma_inv_scale = 0
            gamma_inv_loc   = sim_kwargs['gamma_inv']
            _ext          = "_errorsSampleBeta_Gamma"
            text_error  =  r"$\beta \sim Gamma(.)$"
        ############ Test 2: fix beta, Sample gamma ############
        if test_num == 2:
            beta_scale     = 0
            beta_loc       = sim_kwargs['beta']
            _ext           = "_errorsSampleGamma_Gamma"
            text_error     =  r"$\gamma^{-1} \sim Gamma(.)$"
        ############ Test 3: sample beta, sample gamma ############
        if test_num == 3:
            _ext          =  "_errorsSampleBetaGamma_Gamma"            
            text_error    =  r"$\beta \sim Gamma(.)$" + "\n" + r"$\gamma^{-1} \sim Gamma(.)$"
        
        prob_params = (prob_type, beta_loc, beta_shape, beta_scale, gamma_inv_loc, gamma_inv_shape, gamma_inv_scale) 
    


    return text_error, prob_params, _ext
