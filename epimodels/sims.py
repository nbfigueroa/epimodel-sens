# To read data from xls sheet
import math
from   pandas import DataFrame, read_csv
import pandas as pd 
from   epimodels.utils import *

def loadSimulationParams(sim_num, scenario, plot_data = 1, header = 'SIR'):
    ''' Setups the model parameters and initial conditions for different simulation options
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Baseline case study
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
        number_scaling  = 'million'
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
        number_scaling  = 'million'
        # Load number of infected cases
        Infected, Infected_est = loadYucatanData()    

    
    if sim_num  < 5:        
        figure_title    = r"COVID-19 {header} Model Dynamics [Scenario {scenario:d}] -- {country} -- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
        figure_title    = figure_title.format(header = header, scenario=scenario, country=country, R0=float(r0), beta= beta, gamma = gamma_inv)
    else:    
        # Test Population for Baseline paper
        # Using absolute numbers of population
        # N               = pow(10,5)
        # I0, R0          = 1, 0  
        # number_scaling  = 'million'
        # Using fraction of population
        N               = 1
        I0, R0          = pow(10,-5), 0    
        E0              = 10*I0   # Baseline
        # E0              = I0      # Extreme negative error
        # E0              = 0.001*N # Extreme positive error
        number_scaling  = 'fraction'
        days            = 200
        gamma_inv       = 7.0  
        r0              = 2.31          
        beta            = r0 / gamma_inv
        # sigma_inv       = 5
        sigma_inv       = 4 # New value for paper
        country         = 'Primer'
        x_tick_names    = []
        x_tick_step     = []
        Infected        = np.array([])
        if header == 'SEIR':
            figure_title = r"{header} Model Dynamics -- {country} -- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f}, 1/$\sigma$ = {sigma:1.1f})"
            figure_title    = figure_title.format(header = header, country=country, R0=float(r0), beta= beta, gamma = gamma_inv, sigma = sigma_inv)
        else:
            figure_title    = r"{header} Model Dynamics -- {country} -- ($R_0$={R0:1.3f}, $\beta$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
            figure_title    = figure_title.format(header = header, country=country, R0=float(r0), beta= beta, gamma = gamma_inv)


    # file_extension  = "./results/{country_dir}/VanillaSIR_Scenario{scenario:d}_{country}_{days:d}".format(country_dir= country.lower(), scenario=scenario, country= country, days = days)    

    # Create simulation arguments dictionary with keywords
    sim_kwargs = {}
    sim_kwargs['N']                   = N
    sim_kwargs['days']                = days
    sim_kwargs['beta']                = beta
    sim_kwargs['r0']                  = r0
    sim_kwargs['gamma_inv']           = gamma_inv
    sim_kwargs['sigma_inv']           = sigma_inv
    sim_kwargs['I0']                  = I0
    sim_kwargs['R0']                  = R0
    sim_kwargs['E0']                  = E0
    sim_kwargs['file_extension']      = "./results/{country_dir}/Vanilla{header}_Scenario{scenario:d}_{country}_{days:d}".format(header = header, country_dir= country.lower(), scenario=scenario, country= country, days = days)    
    sim_kwargs['figure_title']        = figure_title
    sim_kwargs['number_scaling']      = number_scaling    
    sim_kwargs['Infected']            = Infected
    sim_kwargs['store_plots']         = 1
    sim_kwargs['do_growth']           = 1
    if len(x_tick_names) > 0:
        sim_kwargs['x_tick_names']    = x_tick_names
        sim_kwargs['x_tick_step']     = x_tick_step

    return sim_kwargs


def getSEIRTestingParams(test_num, prob_type, **sim_kwargs):
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
        sigma_inv_min = 2.2
        sigma_inv_max = 8.0

        ########### Test 1: Sample beta, fix gamma  and sigma ############
        if test_num == 1:
            # this is an indicator that should be max value
            gamma_inv_min = sim_kwargs['gamma_inv']    
            gamma_inv_max = sim_kwargs['gamma_inv'] 
            sigma_inv_min = sim_kwargs['sigma_inv']
            sigma_inv_max = sim_kwargs['sigma_inv']
            _ext          = "_errorsSampleBeta_Uniform"
            text_error  =  r"$\beta \sim \mathcal{U}(0.24,0.42)$"

        ############ Test 2: fix beta and sigma, Sample gamma ############
        if test_num == 2:
            beta_min      = sim_kwargs['beta']
            beta_max      = sim_kwargs['beta']
            sigma_inv_min = sim_kwargs['sigma_inv']
            sigma_inv_max = sim_kwargs['sigma_inv']
            _ext          = "_errorsSampleGamma_Uniform"
            text_error  =  r"$\gamma^{-1} \sim \mathcal{U}(4,10)$"

        ############ Test 3: fix beta, fix gamma, sample beta ############
        if test_num == 3:
            beta_min      = sim_kwargs['beta']
            beta_max      = sim_kwargs['beta']
            gamma_inv_min = sim_kwargs['gamma_inv']    
            gamma_inv_max = sim_kwargs['gamma_inv']
            _ext          = "_errorsSampleSigma_Uniform"            
            text_error    = r"$\sigma^{-1} \sim \mathcal{U}(2.2,8)$"
            #text_error  =  r"$\beta \sim \mathcal{U}(0.24,0.42)$"  + "\n" + r"$\gamma^{-1} \sim \mathcal{U}(4,10)$"
        
        ############ Test 4: Sample beta, gamma, sigma ############
        if test_num == 4:
            _ext    = "_errorsSampleBetaGammaSigma_Uniform"
            text_error    = r"$\beta \sim \mathcal{U}(0.24,0.42)$"  + "\n" + r"$\gamma^{-1} \sim \mathcal{U}(4,10)$" + "\n" + r"$\sigma^{-1} \sim \mathcal{U}(2.2, 8)$"
            
        prob_params = (prob_type, beta_min, beta_max, gamma_inv_min, gamma_inv_max, sigma_inv_min, sigma_inv_max) 

    ######################################################################
    ########### Method 2: Drawing from Gaussian Distributions ############
    ######################################################################
    if prob_type  == 'gaussian':           
        ## For variation on these parameters
        beta                  = sim_kwargs['beta']
        gamma_inv             = sim_kwargs['gamma_inv'] 
        sigma_inv             = sim_kwargs['sigma_inv']
        beta_mean             = beta        
        gamma_inv_mean        = gamma_inv
        sigma_inv_mean        = sigma_inv

        # TODO: Compute the std's with equations from Dave's report
        beta_std              = 0.09
        gamma_inv_std         = 1.73 
        sigma_inv_std         = 1.73
        
        ########### Test 1: Sample beta, fix gamma and sigma ############
        if test_num == 1:
            gamma_inv_std = 0.00
            sigma_inv_std = 0.00
            _ext          = "_errorsSampleBeta_Gaussian"

        ############ Test 2: fix beta and sigma, Sample gamma ############
        if test_num == 2:
            beta_std      = 0.00
            sigma_inv_std = 0.00
            _ext          = "_errorsSampleGamma_Gaussian"

        ############ Test 3: sample sigma, fix beta and gamma ############
        if test_num == 3:
            beta_std      = 0.00
            gamma_inv_std = 0.00
            _ext          =  "_errorsSampleSigma_Gaussian"            
        
        ############ Test 4: Sample beta, gamma, sigma ############
        if test_num == 4:
            _ext          =  "_errorsSampleBetaGammaSigma_Gaussian"
        
        text_error  =  r"$\beta \sim \mathcal{N}(\bar{\beta},%2.2f)$"%beta_std + "\n" +  r"$\gamma^{-1} \sim \mathcal{N}(\gamma^{-1},%2.2f)$"%gamma_inv_std + "\n" + r"$\sigma^{-1} \sim \mathcal{N}(\sigma^{-1},%2.2f)$"%sigma_inv_std
        prob_params = (prob_type, beta_mean, beta_std, gamma_inv_mean, gamma_inv_std, sigma_inv_mean, sigma_inv_std) 
    

    ###################################################################
    ########### Method 3: Drawing from Gamma Distributions ############
    ###################################################################
    if prob_type  == 'gamma':           
        # From equations: \mu=loc, s=shape, \kappa=scale        

        # Old parameters:
        # beta_loc, beta_shape, beta_scale                = 0.22, 10, 0.009
        # gamma_inv_loc, gamma_inv_shape, gamma_inv_scale = 3.8,  10, 0.301

        # New parameters:
        beta_loc, beta_shape, beta_scale                = 0.21, 12, 0.010
        gamma_inv_loc, gamma_inv_shape, gamma_inv_scale = 4.5,  10, 0.25
        sigma_inv_loc, sigma_inv_shape, sigma_inv_scale = 0, 80, 0.05
        

        ########### Test 1: Sample beta, fix gamma and sigma ############
        if test_num == 1:
            gamma_inv_scale = 0
            gamma_inv_loc   = sim_kwargs['gamma_inv']
            sigma_inv_scale = 0
            sigma_inv_loc   = sim_kwargs['sigma_inv']
            _ext          = "_errorsSampleBeta_Gamma"
            text_error  =  r"$\beta \sim Gamma(.)$"

        ############ Test 2: fix beta and sigma, Sample gamma ############
        if test_num == 2:
            beta_scale     = 0
            beta_loc       = sim_kwargs['beta']
            sigma_inv_scale = 0
            sigma_inv_loc  = sim_kwargs['sigma_inv']
            _ext           = "_errorsSampleGamma_Gamma"
            text_error     =  r"$\gamma^{-1} \sim Gamma(.)$"

        ############ Test 3: sample beta, sample gamma ############
        if test_num == 3:
            sigma_inv_scale = 0
            sigma_inv_loc  = sim_kwargs['sigma_inv']
            _ext          =  "_errorsSampleBetaGamma_Gamma"            
            text_error    =  r"$\beta \sim Gamma(.)$" + "\n" + r"$\gamma^{-1} \sim Gamma(.)$"

        ############ Test 4: sample sigma, fix beta and gamma ############
        if test_num == 4:
            beta_scale     = 0
            beta_loc       = sim_kwargs['beta']
            gamma_inv_scale = 0
            gamma_inv_loc   = sim_kwargs['gamma_inv']            
            _ext           = "_errorsSampleSigma_Gamma"
            text_error     = r"$\sigma^{-1} \sim Gamma(.)$"
        
        ############ Test 5: Sample beta, gamma, sigma ############
        if test_num == 5:
            _ext          =  "_errorsSampleBetaGammaSigma_Gamma"            
            text_error    =  r"$\beta \sim Gamma(.)$" + "\n" + r"$\gamma^{-1} \sim Gamma(.)$" + "\n" + r"$\sigma^{-1} \sim Gamma(.)$"
                        
        prob_params = (prob_type, beta_loc, beta_shape, beta_scale, gamma_inv_loc, gamma_inv_shape, gamma_inv_scale, sigma_inv_loc, sigma_inv_shape, sigma_inv_scale) 
    


    #######################################################################
    ########### Method 4: Drawing from LogNormal Distributions ############
    #######################################################################
    if prob_type  == 'log-normal':
        ## For variation on these parameters
        beta_X                  = sim_kwargs['beta']
        gamma_inv_X             = sim_kwargs['gamma_inv'] 
        sigma_inv_X             = sim_kwargs['sigma_inv']
        
        ### TODO: Compute the std's with equations from Dave's report to make it generalizable.. need to define intervals
        beta_std_X              = 0.09 / 3.475
        gamma_inv_std_X         = 1.73  / 2
        sigma_inv_std_X         = 1.73 / 2

        # Convert to logs for log-normal distribution method 1
        beta_mean             = np.log(pow(beta_X,2)/(math.sqrt(pow(beta_X,2) + pow(beta_std_X,2))))       
        beta_std              = math.sqrt(np.log(1 + pow(beta_std_X,2)/pow(beta_X,2)))
        gamma_inv_mean        = np.log(pow(gamma_inv_X,2)/(math.sqrt(pow(gamma_inv_X,2) + pow(gamma_inv_std_X,2))))       
        gamma_inv_std         = math.sqrt(np.log(1 + pow(gamma_inv_std_X,2)/pow(gamma_inv_X,2)))
        sigma_inv_mean        = np.log(pow(sigma_inv_X,2)/(math.sqrt(pow(sigma_inv_X,2) + pow(sigma_inv_std_X,2))))
        sigma_inv_std         = math.sqrt(np.log(1 + pow(sigma_inv_std_X,2)/pow(sigma_inv_X,2)))
        
        # From michigan model estimates
        # r0 = np.random.lognormal(0.31,0.73)
        # gamma = np.random.lognormal(-2.4, 0.73)

        # gamma_inv_mean = np.log(gamma_inv_X)
        # beta_std  = 0.73
        # gamma_inv_std  = 0.73

        ########### Test 1: Sample beta, fix gamma, fix sigma ############
        if test_num == 1:
            gamma_inv_mean = gamma_inv_X
            gamma_inv_std  = 0
            sigma_inv_mean = sigma_inv_X
            sigma_inv_std  = 0
            
            _ext           = "_errorsSampleBeta_LogNormal"

        ############ Test 2: fix beta and sigma, Sample gamma ############
        if test_num == 2:
            beta_mean     = beta_X
            beta_std      = 0
            sigma_inv_mean = sigma_inv_X
            sigma_inv_std  = 0
            _ext          = "_errorsSampleGamma_LogNormal"

        ############ Test 3: sample sigma, fix beta and gamma ############
        if test_num == 3:
            beta_mean     = beta_X
            beta_std      = 0
            gamma_inv_mean = gamma_inv_X
            gamma_inv_std  = 0
            
            _ext          =  "_errorsSampleSigma_LogNormal"            
            
        ############ Test 4: Sample beta, gamma, sigma ############
        if test_num == 4:
            _ext          = "_errorsSampleBetaGammaSigma_LogNormal"
        
        text_error  =  r"$\beta \sim Log-\mathcal{N}(\bar{\beta},%2.3f)$"%beta_std + "\n" +  r"$\gamma^{-1} \sim Log-\mathcal{N}(\gamma^{-1},%2.3f)$"%gamma_inv_std + "\n" + r"$\sigma^{-1} \sim Log-\mathcal{N}(\sigma^{-1},%2.3f)$"%sigma_inv_std
        prob_params = (prob_type, beta_mean, beta_std, gamma_inv_mean, gamma_inv_std, sigma_inv_mean, sigma_inv_std) 
    
    return prob_params, (text_error, _ext)



def getSIRTestingParams(test_num, prob_type, **sim_kwargs):
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

        # TODO: Theoretical range is 1-sigma (from Dave's report)
        # beta_std              = 0.09
        # gamma_inv_std         = 1.73

        # Theoretical range is 2-sigma
        beta_std              = 0.09 * 0.5
        gamma_inv_std         = 1.73 * 0.55
        gamma_inv_std         = 1.73 * 0.5

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
    

    ###################################################################
    ########### Method 3: Drawing from Gamma Distributions ############
    ###################################################################
    if prob_type  == 'gamma':           

        # From equations: \mu=loc, s=shape, \kappa=scale        

        # Old parameters (May draft):
        beta_loc, beta_shape, beta_scale                = 0.22, 10, 0.009
        gamma_inv_loc, gamma_inv_shape, gamma_inv_scale = 3.8,  10, 0.301

        # New parameters (June draft):
        beta_loc, beta_shape, beta_scale                = 0.21, 12, 0.010
        gamma_inv_loc, gamma_inv_shape, gamma_inv_scale = 4.5,  10, 0.25
        

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
    


    #######################################################################
    ########### Method 4: Drawing from LogNormal Distributions ############
    #######################################################################
    if prob_type  == 'log-normal':
        ## For variation on these parameters
        beta_X                  = sim_kwargs['beta']
        gamma_inv_X             = sim_kwargs['gamma_inv'] 
        
        ### TODO: Compute the std's with equations from Dave's report to make it generalizable.. need to define intervals
        beta_std_X              = 0.09 / 3.475
        gamma_inv_std_X         = 1.73  / 2
        # These values are tighter than the ones used for the standard gaussian distribution

        # Convert to logs for log-normal distribution method 1
        beta_mean             = np.log(pow(beta_X,2)/(math.sqrt(pow(beta_X,2) + pow(beta_std_X,2))))       
        beta_std              = math.sqrt(np.log(1 + pow(beta_std_X,2)/pow(beta_X,2)))
        gamma_inv_mean        = np.log(pow(gamma_inv_X,2)/(math.sqrt(pow(gamma_inv_X,2) + pow(gamma_inv_std_X,2))))       
        gamma_inv_std         = math.sqrt(np.log(1 + pow(gamma_inv_std_X,2)/pow(gamma_inv_X,2)))
        
        # From michigan model estimates
        # r0 = np.random.lognormal(0.31,0.73)
        # gamma = np.random.lognormal(-2.4, 0.73)

        # gamma_inv_mean = np.log(gamma_inv_X)
        # beta_std  = 0.73
        # gamma_inv_std  = 0.73

        ########### Test 1: Sample beta, fix gamma, fix sigma ############
        if test_num == 1:
            gamma_inv_mean = gamma_inv_X
            gamma_inv_std  = 0
            _ext           = "_errorsSampleBeta_LogNormal"

        ############ Test 2: fix beta, Sample gamma ############
        if test_num == 2:
            beta_mean     = beta_X
            beta_std      = 0
            _ext          = "_errorsSampleGamma_LogNormal"

        ############ Test 3: sample beta, sample gamma ############
        if test_num == 3:
            _ext          =  "_errorsSampleBetaGamma_LogNormal"            
        
        text_error  =  r"$\beta \sim Log-\mathcal{N}(\bar{\beta},%2.3f)$"%beta_std + "\n" +  r"$\gamma^{-1} \sim Log-\mathcal{N}(\gamma^{-1},%2.3f)$"%gamma_inv_std
        prob_params = (prob_type, beta_mean, beta_std, gamma_inv_mean, gamma_inv_std) 
    
    return prob_params, (text_error, _ext)
