import math
import numpy  as np
from   scipy.optimize import fsolve
from   scipy.signal   import find_peaks
from   scipy          import stats
from   scipy.stats    import gamma as gamma_dist

import matplotlib.pyplot as plt
from   matplotlib import rc
# For beautiful plots
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import statsmodels.api       as sm
from statsmodels.formula.api import ols
from   scipy.stats       import kde
from scipy.interpolate   import UnivariateSpline

import pandas as pd 
import xlsxwriter

###########################################################################################################
#############                          FUNCTIONS FOR DATA LOADING/WRITING                      ############
###########################################################################################################
def loadYucatanData():
    file           = r'./data/covid_yucatan.xlsx'
    case_data      = pd.read_excel(file)
    casos          = case_data['Unnamed: 1']
    muertes        = case_data['Unnamed: 5']
    centinal       = case_data['Unnamed: 3']
    Infected       = np.array(casos[1:])
    Infected_est   = np.array(centinal[1:])

    return Infected, Infected_est


def loadCSSEData(country, plot_data): 
    baseURL         = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    fileName        = "time_series_covid19_confirmed_global.csv"
    data_confirmed  = pd.read_csv(baseURL + fileName) \
                        .drop(['Province/State','Lat', 'Long'], axis=1)
    data_confirmed.set_index('Country/Region', inplace=True)
    country_confirmed     = data_confirmed.loc[country]
    country_confirmed     = country_confirmed.to_numpy()    

    fileName        = "time_series_covid19_recovered_global.csv"
    data_recovered  = pd.read_csv(baseURL + fileName)\
                        .drop(['Province/State','Lat', 'Long'], axis=1)
    data_recovered.set_index('Country/Region', inplace=True)
    country_recovered = data_recovered.loc[country]
    country_recovered = country_recovered.to_numpy()
    
    fileName        = "time_series_covid19_deaths_global.csv"
    data_dead       = pd.read_csv(baseURL + fileName)\
                        .drop(['Province/State','Lat', 'Long'], axis=1)
    data_dead.set_index('Country/Region', inplace=True)
    country_dead    = data_dead.loc[country]
    country_dead    = country_dead.to_numpy()

    data = np.vstack((country_confirmed, country_recovered, country_dead))

    if plot_data:
        T = data[0,:]; R = data[1,:]; D = data[2,:]; I = T - R - D
        t = np.arange(0,len(T),1)        

        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig, ax1 = plt.subplots()    
        fig.suptitle('COVID Cases {country}'.format(country=country),fontsize=20)
        ax1.plot(t, T, 'r.-', lw=2,   label='Total Confirmed (T)')
        ax1.plot(t, I, 'b.-', lw=2,   label='Active (I)')
        ax1.plot(t, R, 'g.-', lw=2,   label='Recovered (R)')
        ax1.plot(t, D, 'm.-', lw=2,   label='Dead (D)')

        
        # Making things beautiful
        ax1.set_xlabel('Time /days', fontsize=20)
        ax1.set_ylabel('Cases', fontsize=20)
        x_tick_numbers  =  np.arange(0, len(T), 14)
        x_tick_names    = ('22 Jan', '5 Feb', '19 Feb', '4 March', '18 March', '1 April', '15 April', '29 April', '13 May', '27 May', '10 June')
        plt.xticks(x_tick_numbers, x_tick_names)
        legend = ax1.legend(fontsize=20)
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax1.spines[spine].set_visible(True)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(20) 
        for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)     
     
        ax1.grid(True, color='k', alpha=0.2, linewidth = 0.25)   
        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
        fig.set_size_inches(22/2, 15/2, forward=True)  
        plt.savefig("./results/COVID_cases_" + country + ".png", bbox_inches='tight')

        plt.show()

    return data


def createResultsfile(basefilename = './results/test', append_name= 'results', test_type='sampling', header="SIR"):
    results_filename = basefilename + '_' + append_name + '.xlsx'
    workbook = xlsxwriter.Workbook(results_filename)
    worksheet = workbook.add_worksheet()

    if header == "SIR":
        if test_type == 'sampling':    
            header = ['beta-mean','beta-Q15.5', 'beta-Q83.5', 'beta-Q2.5','beta-Q97.5',
                      'gamma-inv-mean', 'gamma-inv-Q15.5', 'gamma-inv-Q83.5','gamma-inv-Q2.5','gamma-inv-Q97.5',
                      'R_0-mean',  'R_0-Q15.5','R_0-Q83.5', 'R_0-Q2.5', 'R_0-Q97.5',
                      't_c-mean', 't_c-Q15.5', 't_c-Q83.5', 't_c-Q2.5', 't_c-Q97.5',
                      'I_peak-mean', 'I_peak-Q15.5', 'I_peak-Q83.5', 'I_peak-Q2.5', 'I_peak-Q97.5',
                      'T_end-mean',  'T_end-Q15.5', 'T_end-Q83.5', 'T_end-Q2.5', 'T_end-Q97.5']    

        if test_type == 'varying':    
                    header = ['beta','beta-max', 'beta-max',
                              'gamma_inv', 'gamma_inv-max','gamma_inv-min',
                              'R_0', 'R_0-max', 'R_0-min',
                              't_c', 't_c-max','t_c-min',
                              'I_peak', 'I_peak-max', 'I_peak-min',
                              'T_end', 'T_end-max', 'T_end-min']
    elif header == "SEIR":
        # This is for sampling method.. should be modified
        if test_type == 'sampling':    
            header = ['beta-mean','beta-Q15.5', 'beta-Q83.5', 'beta-Q2.5','beta-Q97.5',
                      'gamma-inv-mean', 'gamma-inv-Q15.5', 'gamma-inv-Q83.5','gamma-inv-Q2.5','gamma-inv-Q97.5',
                      'sigma-inv-mean', 'sigma-inv-Q15.5', 'sigma-inv-Q83.5','sigma-inv-Q2.5','sigma-inv-Q97.5',
                      'R_0-mean',  'R_0-Q15.5','R_0-Q83.5', 'R_0-Q2.5', 'R_0-Q97.5',
                      't_c-mean', 't_c-Q15.5', 't_c-Q83.5', 't_c-Q2.5', 't_c-Q97.5',
                      'I_peak-mean', 'I_peak-Q15.5', 'I_peak-Q83.5', 'I_peak-Q2.5', 'I_peak-Q97.5',
                      'T_end-mean',  'T_end-Q15.5', 'T_end-Q83.5', 'T_end-Q2.5', 'T_end-Q97.5']

        if test_type == 'varying':    
                    header = ['beta','beta-max', 'beta-max',
                              'gamma_inv', 'gamma_inv-max','gamma_inv-min',
                              'sigma_inv', 'sigma_inv-max','sigma_inv-min',
                              'E_0', 'E_0-max','E_0-min',
                              'R_0', 'R_0-max', 'R_0-min',
                              't_c', 't_c-max','t_c-min',
                              'I_peak', 'I_peak-max', 'I_peak-min',
                              'T_end', 'T_end-max', 'T_end-min']

    for i in range(len(header)):
        worksheet.write(0,i, header[i])

    return workbook, worksheet    

########################################################################################################
#############                   Functions FOR ALL MODEL TESTS                               ############
########################################################################################################
def compute_Rt(S, **kwargs):
    N              = kwargs['N']
    
    # If r0 exists in the dictionary
    if 'r0' in kwargs:
        r0   = kwargs['r0']

    # If beta and gamma are given then recompute or overwrite r0 value
    if ('beta' in kwargs) and ('gamma' in kwargs):
        beta   = kwargs['beta']
        gamma  = kwargs['gamma']
        r0     = beta/gamma

    # Compute effective reproductive number curve
    effective_Rt   = r0 * (S/N)

    # Estimations of critical point of epidemic
    tcs_Rt  = np.nonzero(effective_Rt  < 1.0001)
    a = np.array(tcs_Rt)
    if a.size > 0:
        tc_Rt =  tcs_Rt[0][0]
    else: 
        tc_Reff = Nan

    return effective_Rt, tc_Rt

def getCriticalPointsAfterPeak(I):
    """
        Computes t_c, t(I=1000 after t_c), t(I=500 after t_c), t(I=100 after t_c), t(I=10 after t_c)
    """
    tc_, _ = find_peaks(I, distance=1)
    tc = tc_[0]
    I_tc  = I[tc]
    print('Peak Instant. Infected = ', I_tc,'by day=', tc)

    ts_I1000  = np.nonzero(I[tc:-1] < 1001)
    a = np.array(ts_I1000)
    if a.size > 0:
        t_I1000 =  tc + ts_I1000[0][0]
        print('I(t_I1000) = ', I[t_I1000],'by day=', t_I1000)
    else: 
        t_I1000 = []

    ts_I500  = np.nonzero(I[tc:-1] < 501)
    a = np.array(ts_I500)
    if a.size > 0:
        t_I500 =  tc + ts_I500[0][0]
        print('I(t_I500) = ', I[t_I500],'by day=', t_I500)
    else:
        t_I500 = []

    ts_I100  = np.nonzero(I[tc:-1] < 101)
    a = np.array(ts_I100)
    if a.size > 0:
        t_I100 =  tc + ts_I100[0][0]
        print('I(t_I100) = ', I[t_I100],'by day=', t_I100)
    else: 
        t_I100 = []

    ts_I10   = np.nonzero(I[tc:-1] < 11)
    a = np.array(ts_I10)
    if a.size > 0:
        t_I10    =  tc + ts_I10[0][0]
        print('I(t_low) = ', I[t_I10],'by day=', t_I10)
    else: 
        t_I10 = []

    return (tc, t_I100, t_I500, t_I100, t_I10)

def getCriticalPointsDistribution(I_samples, R_samples):

    # Compute Total Cases
    T_samples = I_samples + R_samples

    n_samples, n_days = I_samples.shape

    tc_samples       = np.empty([n_samples, 1])
    Ipeak_samples    = np.empty([n_samples, 1])
    Tend_samples     = np.empty([n_samples, 1])
    
    # TODO: Vectorize
    for ii in range(n_samples):
        tc_, _             = find_peaks(I_samples[ii,:], distance=1)
        # tc, _             = np.argmax(I_samples[ii,:])

        if tc_.size == 0:
            tc = 0
        else:
            tc = tc_[0]   
        tc_samples[ii]    = tc
        Ipeak_samples[ii] = I_samples[ii,tc]
        Tend_samples[ii]  = T_samples[ii,n_days-1]
    
    return (tc_samples, Ipeak_samples, Tend_samples)


def computeStats(X, bound_type='CI', bound_param = [1.96]):
    '''
        Compute mean, median and confidence intervals/quantiles
    '''

    X_bar = np.mean(X, axis=0)    # mean of vector
    X_std = np.std(X, axis=0)     # std of vector
    X_med = np.median(X, axis=0)  # median of vector

    # Computing 95% Confidence Intervals
    if bound_type == 'CI':
        n     = len(X_bar) # number of obs
        # z     = 1.96 # for a 95% CI
        z     = bound_param[0]
        X_lower = X_bar - (z * (X_std/math.sqrt(n)))
        X_upper = X_bar + (z * (X_std/math.sqrt(n)))

    if bound_type == 'Quantiles':    
        # X_lower = np.quantile(X, 0.025, axis = 0)
        # X_upper = np.quantile(X, 0.975, axis = 0)
        X_lower = np.quantile(X, bound_param[0], axis = 0)
        X_upper = np.quantile(X, bound_param[1], axis = 0)   


    return X_bar, X_med, X_std, X_upper, X_lower


def gatherMCstats(S_samples, I_samples, R_samples, E_samples = None, bound_type='CI', bound_param = [1.96]):    
    '''
        Gather stats from MC simulations  
    '''

    T_samples = I_samples + R_samples
    S_mean, S_med, S_std, S_upper, S_lower = computeStats(S_samples, bound_type, bound_param)
    I_mean, I_med, I_std, I_upper, I_lower = computeStats(I_samples, bound_type, bound_param)
    R_mean, R_med, R_std, R_upper, R_lower = computeStats(R_samples, bound_type, bound_param)
    T_mean, T_med, T_std, T_upper, T_lower = computeStats(T_samples, bound_type, bound_param)
    
    if not E_samples is None:
        E_mean, E_med, E_std, E_upper, E_lower = computeStats(E_samples, bound_type, bound_param)

    # Pack values for plotting and analysis

    S_stats          = np.vstack((S_mean, S_med, S_upper, S_lower, S_std))    
    I_stats          = np.vstack((I_mean, I_med, I_upper, I_lower, I_std))    
    R_stats          = np.vstack((R_mean, R_med, R_upper, R_lower, R_std))    
    T_stats          = np.vstack((T_mean, T_med, T_upper, T_lower, T_std))
    if not E_samples is None:
        E_stats      = np.vstack((E_mean, E_med, E_upper, E_lower, E_std))
    
    if not E_samples is None:
        return S_stats, E_stats, I_stats, R_stats, T_stats
    else:
        return S_stats, I_stats, R_stats, T_stats



def printMeanResults(I_stats, R_stats):    
    print('*********   MEAN Results    *********')    
    I_mean = I_stats[0,:]
    R_mean = R_stats[0,:]
    T_mean = I_mean  + R_mean
    tc, t_I100, t_I500, t_I100, t_I10 = getCriticalPointsAfterPeak(I_mean)
    T_tc  = T_mean[tc]
    print('Total Cases @ Peak = ', T_tc,'by day=', tc)
    total_infected     = I_mean[-1]
    print('Infected @ t(end) = ', total_infected)
    total_cases     = T_mean[-1]
    print('Total Cases @ t(end) = ', total_cases)


def sample_SIRparam_distributions(*prob_params):
    '''
        Sample beta and gamma_inv from selected distributions
    '''
    # Sample from Uniform Distributions        
    if prob_params[0] == 'uniform':        
        if prob_params[1] == prob_params[2]:
            beta = prob_params[1]
        else:
            beta = np.random.uniform(prob_params[1],prob_params[2])
        if prob_params[3] == prob_params[4]:
            gamma_inv = prob_params[3]
        else:
            gamma_inv = np.random.uniform(prob_params[3],prob_params[4])

    # Sample from Gaussian Distributions        
    if prob_params[0] == 'gaussian':        
        beta_mean       = prob_params[1]
        beta_std        = prob_params[2]
        gamma_inv_mean  = prob_params[3]
        gamma_inv_std   = prob_params[4]
        
        # Sample from Gaussian Distributions
        beta = 0
        while beta < 0.02:
            beta            = np.random.normal(beta_mean, beta_std)
        gamma_inv = 0    
        while gamma_inv < 0.1:
            gamma_inv       = np.random.normal(gamma_inv_mean, gamma_inv_std)

    # Sample from Gamma Distributions        
    if prob_params[0] == 'gamma':        
        beta_loc        = prob_params[1]
        beta_shape      = prob_params[2]
        beta_scale      = prob_params[3]
        gamma_inv_loc   = prob_params[4]
        gamma_inv_shape = prob_params[5]
        gamma_inv_scale = prob_params[6]

        # Sample from Gamma Distributions
        if beta_scale == 0:
            beta = beta_loc
        else:        
            beta_dist      = gamma_dist(beta_shape, beta_loc, beta_scale)
            beta           = beta_dist.rvs(1)[0]
        if gamma_inv_scale == 0:
            gamma_inv = gamma_inv_loc
        else:            
            gamma_inv_dist = gamma_dist(gamma_inv_shape, gamma_inv_loc, gamma_inv_scale)
            gamma_inv      = gamma_inv_dist.rvs(1)[0]

    # Sample from LogNormal Distributions        
    if prob_params[0] == 'log-normal':
        beta_mean       = prob_params[1]
        beta_std        = prob_params[2]
        gamma_inv_mean  = prob_params[3]
        gamma_inv_std   = prob_params[4]

        # Sample from LogNormal if std not 0
        if beta_std == 0:
            beta = beta_mean
        else:  
            beta  = np.random.lognormal(beta_mean,beta_std)

        # Sample from LogNormal if std not 0
        if gamma_inv_std == 0:
            gamma_inv = gamma_inv_mean
        else:  
            gamma_inv = np.random.lognormal(gamma_inv_mean, gamma_inv_std)

    return beta, gamma_inv



# TODO: Move these to utils.py
def fit1D_KDE(x):
    # Use scipy.stats class to fit the bandwith
    tc_kde = stats.gaussian_kde(x.T, bw_method = 'silverman')    
    stdev = np.sqrt(tc_kde.covariance)[0, 0]

    # using statsmodels kde to compute quantiles
    tc_kde_sm = sm.nonparametric.KDEUnivariate(x)    
    tc_kde_sm.fit()
    bw = tc_kde_sm.bw
    print('KDE std-dev:', stdev, ' bw:', bw)
    tc_kde_sm.fit(bw=np.max(np.array([stdev,bw])))
    icdf_spl = UnivariateSpline(np.linspace(0, 1, num = tc_kde_sm.icdf.size), tc_kde_sm.icdf)    

  
    fun_kde      = lambda x: tc_kde_sm.evaluate(x)
    fun_kde_icdf = lambda x: icdf_spl(x)

    return  fun_kde, fun_kde_icdf    


def hyperplane_similarity(w_1,b_1,w_2,b_2, sim_type = 'sim'):
    """
        Equation for Hyper-plane similarity measure 
        https://math.stackexchange.com/questions/2124611/on-a-measure-of-similarity-between-two-hyperplanes

        Equation for Dihedral angle
        https://en.wikipedia.org/wiki/Dihedral_angle

    """
    if sim_type == 'sim':
        # Original version    
        d = (np.linalg.norm(w_1)*np.linalg.norm(w_2)) - abs(np.dot(w_1, w_2)) + abs(b_1 - b_2)
        
        # Normalized version
        d = 1 - abs(np.dot(w_1, w_2))/(np.linalg.norm(w_1)*np.linalg.norm(w_2)) + abs(b_1 - b_2)
    else:    
        # Dihedral angle (angle between two intersecting hyper-planes.. are they?)
        n_1  = np.random.randn(2)
        n_1 -= n_1.dot(w_1)  * (w_1 / np.linalg.norm(w_1)**2)
        n_1 /= np.linalg.norm(n_1)
        
        n_2  = np.random.randn(2)
        n_2 -= n_2.dot(w_2)  * (w_2 / np.linalg.norm(w_2)**2)
        n_2 /= np.linalg.norm(n_2)

        d = math.acos(abs(np.dot(n_1,n_2))/abs(np.linalg.norm(n_1) * np.linalg.norm(n_2)))
        # d_ = math.acos(abs(np.dot(w_1,w_2))/abs(np.linalg.norm(w_1) * np.linalg.norm(w_2)))

    return d


def computeCriticalPointsStats(SIR_params, CO_samples, plot_options, **kwargs):

    plot_data_quant, plot_regress_lines, do_histograms, do_contours, do_mask = plot_options

    beta_samples      = SIR_params[:,0]
    gamma_inv_samples = SIR_params[:,1]
    tc_samples, Ipeak_samples, Tend_samples = CO_samples

    if 'Compartment' in kwargs['figure_title']:        
        model = StochasticCambridge(1)
        R0_samples = [model.compute_r0(sample) for sample in SIR_params]
    else:
        R0_samples = beta_samples * gamma_inv_samples

    if 'SEIR' in kwargs['figure_title']:
        sigma_inv_samples = SIR_params[:,2]

    ############################################################################################################
    #######  Compute Descriptive Stats for each critical point distributions (t_c, I_peak, T_end and R0) #######
    ############################################################################################################

    ####################################
    ##### Stats on inputs to model #####
    ####################################
    beta_bar, beta_med, beta_std, beta_upper95, beta_lower95 = computeStats(beta_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, beta_upper68, beta_lower68 = computeStats(beta_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    beta_skew = stats.skew(beta_samples)    
    print('Mean beta=',beta_bar, ' Med beta=', beta_med, 'Skew beta=', beta_skew)

    gamma_inv_bar, gamma_inv_med, gamma_inv_std, gamma_inv_upper95, gamma_inv_lower95 = computeStats(gamma_inv_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, gamma_inv_upper68, gamma_inv_lower68 = computeStats(gamma_inv_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    gamma_inv_skew = stats.skew(gamma_inv_samples)    
    print('Mean gamma_inv=',gamma_inv_bar, ' Med gamma_inv=', gamma_inv_med, 'Skew gamma_inv=', gamma_inv_skew)

    if 'SEIR' in kwargs['figure_title']:
        sigma_inv_bar, sigma_inv_med, sigma_inv_std, sigma_inv_upper95, sigma_inv_lower95 = computeStats(sigma_inv_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
        _, _, _, sigma_inv_upper68, sigma_inv_lower68 = computeStats(sigma_inv_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
        sigma_inv_skew = stats.skew(sigma_inv_samples)    
        print('Mean sigma_inv=',sigma_inv_bar, ' Med sigma_inv=', sigma_inv_med, 'Skew sigma_inv=', sigma_inv_skew)


    R0_bar, R0_med, R0_std, R0_upper95, R0_lower95 = computeStats(R0_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, R0_upper68, R0_lower68 = computeStats(R0_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    R0_skew = stats.skew(R0_samples)    
    print('Mean tc=',R0_bar, ' Med tc=', R0_med, 'Skew tc=', R0_skew)

    #####################################
    ##### Stats on outputs of model #####
    #####################################
    tc_bar, tc_med, tc_std, tc_upper95, tc_lower95 = computeStats(tc_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, tc_upper68, tc_lower68 = computeStats(tc_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    tc_skew = stats.skew(tc_samples)    
    print('Mean tc=',tc_bar, ' Med tc=', tc_med, 'Skew tc=', tc_skew)

    Ipeak_bar, Ipeak_med, Ipeak_std, Ipeak_upper95, Ipeak_lower95 = computeStats(Ipeak_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, Ipeak_upper68, Ipeak_lower68 = computeStats(Ipeak_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    Ipeak_skew = stats.skew(Ipeak_samples)
    print('Mean Ipeak=',Ipeak_bar, ' Med Ipeak=', Ipeak_med, 'Skew Ipeak=', Ipeak_skew)

    Tend_bar, Tend_med, Tend_std, Tend_upper95, Tend_lower95 = computeStats(Tend_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, Tend_upper68, Tend_lower68 = computeStats(Tend_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    Tend_skew = stats.skew(Tend_samples)
    print('Mean Tend=',Tend_bar, ' Med Tend=', Tend_med, 'Skew Tend=', Tend_skew)

    #################################################################################
    ####### Fit kde to each critical point and compute stats of distributions #######
    #################################################################################

    # Fit kde to Tend_samples
    tc_kde, tc_kde_icdf = fit1D_KDE(tc_samples)
    x_vals_tc      = np.linspace(np.min(tc_samples),np.max(tc_samples), 100)
    tc_kde_pdf     = tc_kde(x_vals_tc)
    tc_kde_median  = tc_kde_icdf(0.50)
    tc_kde_lower95 = tc_kde_icdf(0.025) 
    tc_kde_upper95 = tc_kde_icdf(0.975) 
    tc_kde_lower68 = tc_kde_icdf(0.155) 
    tc_kde_upper68 = tc_kde_icdf(0.835) 
    print('t_c:', tc_kde_median, tc_kde_lower95, tc_kde_upper95, tc_kde_lower68, tc_kde_upper68)

    # Fit kde to Tend_samples
    Ipeak_kde, Ipeak_kde_icdf = fit1D_KDE(Ipeak_samples)
    x_vals_Ipeak      = np.linspace(np.min(Ipeak_samples),np.max(Ipeak_samples), 100)
    Ipeak_kde_pdf     = Ipeak_kde(x_vals_Ipeak)
    Ipeak_kde_median  = Ipeak_kde_icdf(0.50)
    Ipeak_kde_lower95 = Ipeak_kde_icdf(0.025) 
    Ipeak_kde_upper95 = Ipeak_kde_icdf(0.975) 
    Ipeak_kde_lower68 = Ipeak_kde_icdf(0.155) 
    Ipeak_kde_upper68 = Ipeak_kde_icdf(0.835) 
    print('Ipeak:', Ipeak_kde_median, Ipeak_kde_lower95, Ipeak_kde_upper95, Ipeak_kde_lower68, Ipeak_kde_upper68)

    # Fit kde to Tend_samples
    Tend_kde, Tend_kde_icdf = fit1D_KDE(Tend_samples)
    x_vals_Tend = np.linspace(np.min(Tend_samples),np.max(Tend_samples), 100)
    Tend_kde_pdf = Tend_kde(x_vals_Tend)
    Tend_kde_median  = Tend_kde_icdf(0.50)
    Tend_kde_lower95 = Tend_kde_icdf(0.025) 
    Tend_kde_upper95 = Tend_kde_icdf(0.975) 
    Tend_kde_lower68 = Tend_kde_icdf(0.155) 
    Tend_kde_upper68 = Tend_kde_icdf(0.835) 
    print('Tend:', Tend_kde_median, Tend_kde_lower95, Tend_kde_upper95, Tend_kde_lower68, Tend_kde_upper68)

    # Store stats in worksheet
    worksheet        = kwargs['worksheet']
    row_num          = kwargs['row_num']
    beta_stats       = [beta_bar, beta_lower68, beta_upper68, beta_lower95, beta_upper95]
    gamma_inv_stats  = [gamma_inv_bar, gamma_inv_lower68,  gamma_inv_upper68, gamma_inv_lower95, gamma_inv_upper95]
    if 'SEIR' in kwargs['figure_title']:
        sigma_inv_stats  = [sigma_inv_bar, sigma_inv_lower68,  sigma_inv_upper68, sigma_inv_lower95, sigma_inv_upper95]
    R0_stats         = [R0_bar, R0_lower68, R0_upper68, R0_lower95, R0_upper95]
    tc_stats         = [tc_bar, tc_kde_lower68, tc_kde_upper68, tc_kde_lower95, tc_kde_upper95]
    Ipeak_stats      = [Ipeak_bar, Ipeak_kde_lower68, Ipeak_kde_upper68, Ipeak_kde_lower95, Ipeak_kde_upper95]
    Tend_stats       = [Tend_bar, Tend_kde_lower68, Tend_kde_upper68, Tend_kde_lower95, Tend_kde_upper95]

    worksheet.write_row(row_num, 0,  beta_stats)
    worksheet.write_row(row_num, 5,  gamma_inv_stats)
    if 'SEIR' in kwargs['figure_title']:
        worksheet.write_row(row_num, 10,  sigma_inv_stats)
        worksheet.write_row(row_num, 15, R0_stats)    
        worksheet.write_row(row_num, 20, tc_stats)
        worksheet.write_row(row_num, 25, Ipeak_stats)
        worksheet.write_row(row_num, 30, Tend_stats)
    else:
        worksheet.write_row(row_num, 10, R0_stats)    
        worksheet.write_row(row_num, 15, tc_stats)
        worksheet.write_row(row_num, 20, Ipeak_stats)
        worksheet.write_row(row_num, 25, Tend_stats)

    ####################################################################
    #### Plot histograms of t_c, I_peak and T_end vs. param-vector #####
    ####################################################################
    if do_histograms:
        #### Plot for t_c ####
        fig, (ax0,ax1,ax2)   = plt.subplots(1,3, constrained_layout=True)
        bin_size = 30    

        # Histogram
        count, bins, ignored = ax0.hist(tc_samples, bin_size, density=True, color='r', alpha = 0.35, edgecolor='k' )
        
        # Plot kde curve and quantile stats    
        # ax0.plot(x_vals_tc,tc_kde_pdf,'k', lw=1, label=r"kde")
        ax0.plot(x_vals_tc,tc_kde_pdf,'k', lw=2)
        ax0.plot([tc_bar[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r', lw=2,label=r"mean")        
        ax0.plot([tc_kde_median]*10,np.linspace(0,count[np.argmax(count)], 10),'k--', lw=2,label=r"med")
        ax0.plot([tc_kde_lower95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=3, label=r"Q[95\%]")
        ax0.plot([tc_kde_upper95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=3)
        ax0.plot([tc_kde_lower68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=3, label=r"Q[68\%]")
        ax0.plot([tc_kde_upper68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=3)

        # Plot raw mean and quantile stats        
        if plot_data_quant:
            ax0.plot([tc_lower95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r:', lw=1.5, label=r"Q[95\%]")
            ax0.plot([tc_upper95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r:', lw=1.5)
            ax0.plot([tc_lower68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r-.', lw=1.5, label=r"Q[68\%]")
            ax0.plot([tc_upper68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r-.', lw=1.5)

        # ax0.set_title(r"Critical point $t_c$", fontsize=20)
        ax0.grid(True, alpha=0.3)
        ax0.set_xlabel(r"Critical point $t_c$", fontsize=20)
        legend = ax0.legend(fontsize=17, loc='upper right')
        legend.get_frame().set_alpha(0.5)
        for tick in ax0.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        for tick in ax0.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 

        #########################        
        #### Plot for I_peak ####
        #########################

        ### Histogram and stats for I_peak
        count, bins, ignored = ax1.hist(Ipeak_samples, bin_size, density=True, color='g', alpha = 0.35, edgecolor='k')
        
        # Plot kde curve and quantile stats 
        
        # ax1.plot(x_vals_Ipeak,Ipeak_kde_pdf,'k', lw=1, label=r"kde")
        ax1.plot(x_vals_Ipeak,Ipeak_kde_pdf,'k', lw=2)
        ax1.plot([Ipeak_bar[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g', lw=2, label=r"mean")      
        ax1.plot([Ipeak_kde_median]*10,np.linspace(0,count[np.argmax(count)], 10),'k--', lw=2,label=r"med")
        ax1.plot([Ipeak_kde_lower95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=3, label=r"Q[95\%]")
        ax1.plot([Ipeak_kde_upper95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=3)
        ax1.plot([Ipeak_kde_lower68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=3, label=r"Q[68\%]")
        ax1.plot([Ipeak_kde_upper68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=3)

        # Plot raw median and quantile stats        
        if plot_data_quant:
            ax1.plot([Ipeak_lower95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g:', lw=1.5, label=r"Q[95\%]")
            ax1.plot([Ipeak_upper95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g:', lw=1.5)
            ax1.plot([Ipeak_lower68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g-.', lw=1.5, label=r"Q[68\%]")
            ax1.plot([Ipeak_upper68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g-.', lw=1.5)

        # ax1.set_title(r"Peak Infected $I_{peak}$", fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel(r"Peak Infected $I_{peak}$", fontsize=20)
        legend = ax1.legend(fontsize=17, loc='upper right')
        legend.get_frame().set_alpha(0.5)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 

        ###########################        
        #### Plot for T(t=end) ####
        ###########################

        ### Histogram and stats for T(t=end)
        count, bins, ignored = ax2.hist(Tend_samples, bin_size, density=True, color='b', alpha = 0.35, edgecolor='k')   

        # Plot kde curve and quantile stats    
        
        # ax2.plot(x_vals_Tend,Tend_kde_pdf,'k', lw=1, label=r"kde")    
        ax2.plot(x_vals_Tend,Tend_kde_pdf,'k', lw=2)  
        ax2.plot([Tend_kde_median]*10,np.linspace(0,count[np.argmax(count)], 10),'k--', lw=2,label=r"med")
        ax2.plot([Tend_bar[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b', lw=2,label=r"mean")    
        ax2.plot([Tend_kde_lower95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=3, label=r"Q[95\%]")
        ax2.plot([Tend_kde_upper95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=3)
        ax2.plot([Tend_kde_lower68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=3, label=r"Q[68\%]")
        ax2.plot([Tend_kde_upper68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=3)

        # Plot raw median and quantile stats 
        
        if plot_data_quant:
            ax2.plot([Tend_lower95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b:', lw=1.5, label=r"Q[95\%]")
            ax2.plot([Tend_upper95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b:', lw=1.5)
            ax2.plot([Tend_lower68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b-.', lw=1.5, label=r"Q[68\%]")
            ax2.plot([Tend_upper68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b-.', lw=1.5)

        # ax2.set_title(, fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel(r"Total cases $T_{\rm end}$", fontsize=20)
        legend = ax2.legend(fontsize=17, loc='upper left')
        legend.get_frame().set_alpha(0.5)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 

        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
        fig.set_size_inches(27/2, 9/2, forward=True)
        if kwargs['store_plots']:
            plt.savefig(kwargs['file_extension'] + "_CriticalPointsHistograms.png", bbox_inches='tight')




    #############################################################################
    ####### Fit regressive models of model parameters vs. R0 and outcomes #######
    #############################################################################
    # Variables for Critical Point samples
    t_tc = tc_samples[:,0]
    t_Ipeak = Ipeak_samples[:,0]
    t_Tend = Tend_samples[:,0]

    # Learn linear regressor between params and R0
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    regr = linear_model.LinearRegression()
    X = SIR_params[:, 0:2]; 
    Y = R0_samples
    regr.fit(X, Y)
    Y_pred = np.empty([1,len(Y)])
    for ii in range(len(Y)):
        Y_pred[0,ii] =  regr.predict([X[ii,:]])
    print('R0 MSE: ', mean_squared_error(Y_pred.T, Y))
    print('R0 R2: ',  r2_score(Y_pred.T, Y))
    print('R0 Coeff:', regr.coef_, 'R0 Intercept:', regr.intercept_)
    print('R0 min/max=',np.min(R0_samples), np.max(R0_samples))

    # Learn linear regressor between params and tc
    regr_tc = linear_model.LinearRegression()
    Y = tc_samples
    regr_tc.fit(X, Y)
    for ii in range(len(Y)):
        Y_pred[0,ii] =  regr_tc.predict([X[ii,:]])

    print('t_c MSE: ', mean_squared_error(Y_pred.T, Y))
    print('t_c R2: ',  r2_score(Y_pred.T, Y))
    print('t_c Coeff:', regr_tc.coef_, 't_c Intercept:', regr_tc.intercept_)

    regr_Ipeak = linear_model.LinearRegression()
    Y = Ipeak_samples
    regr_Ipeak.fit(X, Y)    
    for ii in range(len(Y)):
        Y_pred[0,ii] =  regr_Ipeak.predict([X[ii,:]])

    print('Ipeak MSE: ', mean_squared_error(Y_pred.T, Y))
    print('Ipeak R2: ',  r2_score(Y_pred.T, Y))     
    print('Ipeak Coeff:', regr_Ipeak.coef_, 't_c Intercept:', regr_Ipeak.intercept_)

    regr_Tend = linear_model.LinearRegression()
    Y = Tend_samples    
    regr_Tend.fit(X, Y)
    for ii in range(len(Y)):
        Y_pred[0,ii] =  regr_Tend.predict([X[ii,:]])
    
    print('Tend MSE: ', mean_squared_error(Y_pred.T, Y))
    print('Tend R2: ',  r2_score(Y_pred.T, Y))        
    print('Tend Coeff:', regr_Tend.coef_, 'Tend Intercept:', regr_Tend.intercept_)

    # Compute similarities (Could make this a matrix)
    dist_types = ['sim', 'angle']
    for dist_type in dist_types:
        d_R0tc      = hyperplane_similarity(regr.coef_,regr.intercept_,regr_tc.coef_[0],regr_tc.intercept_[0], dist_type)
        d_R0IPeak   = hyperplane_similarity(regr.coef_,regr.intercept_,regr_Ipeak.coef_[0],regr_Ipeak.intercept_[0], dist_type)
        d_R0Tend    = hyperplane_similarity(regr.coef_,regr.intercept_,regr_Tend.coef_[0],regr_Tend.intercept_[0], dist_type)
        d_tcIpeak   = hyperplane_similarity(regr_tc.coef_[0],regr_tc.intercept_[0],regr_Ipeak.coef_[0],regr_Ipeak.intercept_[0], dist_type)
        d_tcTend    = hyperplane_similarity(regr_tc.coef_[0],regr_tc.intercept_[0],regr_Tend.coef_[0],regr_Tend.intercept_[0], dist_type)
        d_IpeakTend = hyperplane_similarity(regr_Ipeak.coef_[0],regr_Ipeak.intercept_[0],regr_Tend.coef_[0],regr_Tend.intercept_[0], dist_type)

        print(dist_type, ' d_R0tc = ',d_R0tc, ' d_R0IPeak=', d_R0IPeak, ' d_R0Tend=', d_R0Tend)
        print(dist_type, 'd_tcIpeak = ',d_tcIpeak, ' d_tcTend=', d_tcTend, ' d_IpeakTend=', d_IpeakTend)

    if do_mask:
        # Options for masking data
        mask_type = '95'

        # Masked point samples for 95% of outcomes            
        if mask_type == '95':
            do_95 = 1
            idx_tc       = np.nonzero(np.logical_and(t_tc > tc_kde_lower95, t_tc < tc_kde_upper95))
            masked_tc    = t_tc[idx_tc]
            idx_Ipeak    = np.nonzero(np.logical_and(t_Ipeak > Ipeak_kde_lower95, t_Ipeak < Ipeak_kde_upper95))
            masked_Ipeak = t_Ipeak[idx_Ipeak]
            idx_Tend     = np.nonzero(np.logical_and(t_Tend > Tend_kde_lower95, t_Tend < Tend_kde_upper95))
            masked_Tend  = t_Tend[idx_Tend]

        # Masked point samples for 68% of outcomes
        if mask_type == '68':
            do_95 = 0            
            idx_tc       = np.nonzero(np.logical_and(t_tc > tc_kde_lower68, t_tc < tc_kde_upper68))
            masked_tc    = t_tc[idx_tc]
            idx_Ipeak    = np.nonzero(np.logical_and(t_Ipeak > Ipeak_kde_lower68, t_Ipeak < Ipeak_kde_upper68))
            masked_Ipeak = t_Ipeak[idx_Ipeak]
            idx_Tend     = np.nonzero(np.logical_and(t_Tend > Tend_kde_lower68, t_Tend < Tend_kde_upper68))
            masked_Tend  = t_Tend[idx_Tend]        

        # Masked point samples for r0 slice
        if mask_type == 'R0':
            do_95        = -1           

            R0_nom = 2.3; R0_err = 0.20
            R0_min = R0_nom - R0_nom*R0_err
            R0_max = R0_nom + R0_nom*R0_err

            idx_R0       = np.nonzero(np.logical_and(R0_samples > R0_min, R0_samples < R0_max))
            idx_tc = idx_R0; idx_Ipeak = idx_R0; idx_Tend = idx_R0
            masked_tc    = t_tc[idx_tc]
            masked_Ipeak = t_Ipeak[idx_Ipeak]
            masked_Tend  = t_Tend[idx_Tend]        

            print('R0 Error band:' , (R0_max - R0_min)/R0_nom)

            maskedtc_bar, maskedtc_med, tc_std, maskedtc_upper95, maskedtc_lower95 = computeStats(masked_tc, bound_type='Quantiles', bound_param = [0.025, 0.975])
            print('MASKED:: Mean tc=',maskedtc_bar, ' Med tc=', maskedtc_med, 'Up.Q tc=', maskedtc_upper95, 'Low.Q tc=', maskedtc_lower95)
            tc_error = (maskedtc_upper95 - maskedtc_lower95)/maskedtc_bar
            print('Error band:' , tc_error)

            maskedIpeak_bar, maskedIpeak_med, Ipeak_std, maskedIpeak_upper95, maskedIpeak_lower95 = computeStats(masked_Ipeak, bound_type='Quantiles', bound_param = [0.025, 0.975])
            print('MASKED:: Mean Ipeak=',maskedIpeak_bar, ' Med Ipeak=', maskedIpeak_med, 'Up.Q Ipeak=', maskedIpeak_upper95, 'Low.Q Ipeak=', maskedIpeak_lower95)
            Ipeak_error = (maskedIpeak_upper95 - maskedIpeak_lower95)/maskedIpeak_bar
            print('Error band:' , Ipeak_error)

            maskedTend_bar, maskedTend_med, Tend_std, maskedTend_upper95, maskedTend_lower95 = computeStats(masked_Tend, bound_type='Quantiles', bound_param = [0.025, 0.975])
            print('MASKED:: Mean Tend=', maskedTend_bar, ' Med Tend=', maskedTend_med, 'Up.Q Tend=', maskedTend_upper95, 'Low.Q Tend=', maskedTend_lower95)
            Tend_error = (maskedTend_upper95 - maskedTend_lower95)/maskedTend_bar
            print('Error band:' , Tend_error)


    #################################################################################
    ####     Plot scatter of paramaters vs critical points when both are sampled  ###
    #################################################################################
    beta_std      = np.std(beta_samples, axis=0)
    gamma_inv_std = np.std(gamma_inv_samples, axis=0)
    if do_contours:
        if beta_std != 0 and gamma_inv_std != 0:
            ###############################################################################################
            #### Contour plot of 2D gaussian kde Param distribution with regression hyper-plane for R0 ####
            ###############################################################################################
            x = beta_samples
            y = gamma_inv_samples
            xmin = np.min(beta_samples); xmax = np.max(beta_samples); 
            ymin = np.min(gamma_inv_samples); ymax = np.max(gamma_inv_samples);        
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([beta_samples, gamma_inv_samples])
            kde_2d = stats.gaussian_kde(values)
            f = np.reshape(kde_2d(positions).T, xx.shape)

            fig00, ax00 = plt.subplots()     
            
            if plot_regress_lines:
                cset = ax00.contour(xx, yy, f, colors='darkblue', levels = 10, alpha = 0.25)    
                dim, N = positions.shape
                Y = np.empty([1,N])
                for ii in range(N):
                    y_regr =  regr.predict([[positions[0,ii],positions[1,ii]]])
                    # y_regr =  regr.predict([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii]]])
                    # y_regr =  regr.predict([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii], positions[0,ii]**2, positions[1,ii]**2]])
                    Y[0,ii]  = y_regr[0]
                f_R0 = np.reshape(Y.T, xx.shape)
                cset = ax00.contour(xx, yy, f_R0, colors='k', levels = 10, alpha = 0.75)
                ax00.clabel(cset, inline=1, fontsize=10)                
            else:
                cset = ax00.contour(xx, yy, f, colors='k', levels = 15, alpha = 0.85)
                ax00.clabel(cset, inline=1, fontsize=10)        

            cax = ax00.scatter(beta_samples, gamma_inv_samples, c=R0_samples,  cmap='tab20c', alpha = 0.85, s= 10)
            ax00.set_xlabel(r"$\beta$", fontsize=20)
            ax00.set_ylabel(r"$\gamma^{-1}$", fontsize=20)
            for tick in ax00.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
            for tick in ax00.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)     
            ax00.grid(True, alpha=0.3)
            ax00.set_title(r"Distribution of $\beta,\gamma^{-1}$ vs. $R_{0}$", fontsize=20)
            cbar = fig00.colorbar(cax, ax=ax00, orientation='vertical')
            cbarlabels = np.around(np.linspace(np.min(R0_samples), np.max(R0_samples), num = 20, endpoint=True),decimals=2)
            cbar.set_ticks(cbarlabels)
            cbar.set_ticklabels(cbarlabels)
            fig00.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
            # fig00.tight_layout()
            fig00.set_size_inches(29/2 * 0.35, 9/2, forward=True)

            if kwargs['store_plots']:
                if plot_regress_lines:
                    plt.savefig(kwargs['file_extension'] + "_ParamSamplesContours_regress.png", bbox_inches='tight')
                else: 
                    plt.savefig(kwargs['file_extension'] + "_ParamSamplesContours.png", bbox_inches='tight')

            ##########################################################################################
            #### Contour plots of 2D gaussian kde Param distribution with regression hyper-plane  ####
            ##########################################################################################       
            fig0, (ax01,ax02,ax03) = plt.subplots(1,3)        
            
            if plot_regress_lines:
                cset = ax01.contour(xx, yy, f, colors='darkblue', levels = 10, alpha = 0.25)
                dim, N = positions.shape
                Y = np.empty([1,N])
                for ii in range(N):
                    y_regr =  regr_tc.predict([[positions[0,ii],positions[1,ii]]])
                    # y_regr =  regr_tc.predict([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii]]])
                    # y_regr =  regr_tc.predict(([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii], positions[0,ii]**2, positions[1,ii]**2]]))
                    Y[0,ii]  = y_regr[0]
                f_tc = np.reshape(Y.T, xx.shape)
                cset = ax01.contour(xx, yy, f_tc, colors='k', levels = 10, alpha = 0.75)
                ax01.clabel(cset, inline=1, fontsize=10)          
                # cax = ax01.scatter(x[idx_tc], y[idx_tc], c=masked_tc,  cmap='RdBu', alpha = 0.35, s= 10)
                cax = ax01.scatter(x[idx_tc], y[idx_tc], c=masked_tc,  cmap='tab20c', alpha = 0.55, s= 10)
            else:
                cax = ax01.scatter(x, y, c='w', edgecolor='k', alpha = 0.35, s= 10)
                cset = ax01.contour(xx, yy, f, colors='k', levels = 15, alpha = 0.85)
                ax01.clabel(cset, inline=1, fontsize=10)                
                # cax = ax01.scatter(x[idx_tc], y[idx_tc], c=masked_tc,  cmap='RdBu', alpha = 0.35, s= 10)
                cax = ax01.scatter(x[idx_tc], y[idx_tc], c=masked_tc,  cmap='tab20c', alpha = 0.85, s= 10)

                print("MASKED t_c: min beta=", np.argmin(x[idx_tc]), "max beta=", np.argmax(x[idx_tc]))
                print("MASKED t_c: min gamma=", np.argmin(y[idx_tc]), "max gamma=", np.argmax(y[idx_tc]))


            ax01.set_xlabel(r"$\beta$", fontsize=20)
            ax01.set_ylabel(r"$\gamma^{-1}$", fontsize=20)
            for tick in ax01.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
            for tick in ax01.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
            ax01.grid(True, alpha=0.3)
            if do_95 == 1:
                ax01.set_title(r'$95\% t_{c}$ Values')
            elif do_95 == 0:
                ax01.set_title(r'$ 68\% t_{c}$ Values')
            elif do_95 == -1:
                ax01.set_title(r'$\mathcal{R}_0$ Slice $t_{c}$ Values')
            fig0.colorbar(cax, ax=ax01, shrink=0.9)

            if plot_regress_lines:
                cset = ax02.contour(xx, yy, f, colors='darkblue', levels = 10, alpha = 0.25)
                # ax02.clabel(cset, inline=1, fontsize=10)        
                Y = np.empty([1,N])
                for ii in range(N):
                    y_regr =  regr_Ipeak.predict([[positions[0,ii],positions[1,ii]]])
                    # y_regr =  regr_Ipeak.predict([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii]]])
                    # y_regr =  regr_Ipeak.predict(([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii], positions[0,ii]**2, positions[1,ii]**2]]))
                    Y[0,ii]  = y_regr[0]
                f_Ipeak = np.reshape(Y.T, xx.shape)
                cset = ax02.contour(xx, yy, f_Ipeak, colors='k', levels = 15, alpha = 0.75)
                ax02.clabel(cset, inline=1, fontsize=10)          
                # cax = ax02.scatter(x[idx_Ipeak], y[idx_Ipeak], c=masked_Ipeak,  cmap='PiYG', alpha = 0.35, s= 10)
                cax = ax02.scatter(x[idx_Ipeak], y[idx_Ipeak], c=masked_Ipeak,  cmap='tab20c', alpha = 0.55, s= 10)
            else:
                cax = ax02.scatter(x, y, c='w', edgecolor='k', alpha = 0.35, s= 10)
                cset = ax02.contour(xx, yy, f, colors='k', levels = 15, alpha = 0.85)
                ax02.clabel(cset, inline=1, fontsize=10)        
                # cax = ax02.scatter(x[idx_Ipeak], y[idx_Ipeak], c=masked_Ipeak,  cmap='PiYG', alpha = 0.35, s= 10)
                cax = ax02.scatter(x[idx_Ipeak], y[idx_Ipeak], c=masked_Ipeak,  cmap='tab20c', alpha = 0.85, s= 10)
                print("MASKED I_peak: min beta=", np.argmin(x[idx_Ipeak]), "max beta=", np.argmax(x[idx_Ipeak]))
                print("MASKED I_peak: min gamma=", np.argmin(y[idx_Ipeak]), "max gamma=", np.argmax(y[idx_Ipeak]))
            
            ax02.set_xlabel(r"$\beta$", fontsize=20)
            ax02.set_ylabel(r"$\gamma^{-1}$", fontsize=20)
            for tick in ax02.xaxis.get_major_ticks(): 
                tick.label.set_fontsize(15) 
            for tick in ax02.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
            ax02.grid(True, alpha=0.3)
            if do_95 == 1:
                ax02.set_title(r'$95\% I_{peak}$ Values')
            elif do_95 == 0:
                ax02.set_title(r'$68\% I_{peak}$ Values')
            elif do_95 == -1:
                ax02.set_title(r'$\mathcal{R}_0$ Slice $I_{peak}$ Values')
            fig0.colorbar(cax, ax=ax02, shrink=0.9)

            ############ Param samples vs. Tend #############            
            if plot_regress_lines:
                cset = ax03.contour(xx, yy, f, colors='darkblue', levels = 10, alpha = 0.25)
                # ax02.clabel(cset, inline=1, fontsize=10)                        
                Y = np.empty([1,N])
                for ii in range(N):
                    y_regr =  regr_Tend.predict([[positions[0,ii],positions[1,ii]]])
                    # y_regr =  regr_Tend.predict([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii]]])
                    # y_regr =  regr_Tend.predict(([[positions[0,ii],positions[1,ii], positions[0,ii]*positions[1,ii], positions[0,ii]**2, positions[1,ii]**2]]))
                    Y[0,ii]  = y_regr[0]
                f_Tend = np.reshape(Y.T, xx.shape)
                cset = ax03.contour(xx, yy, f_Tend, colors='k', levels = 10, alpha = 0.75)
                ax02.clabel(cset, inline=1, fontsize=10)
                # cax = ax03.scatter(x[idx_Tend], y[idx_Tend], c=masked_Tend,  cmap='bwr', alpha = 0.35, s= 10)
                cax = ax03.scatter(x[idx_Tend], y[idx_Tend], c=masked_Tend,  cmap='tab20c', alpha = 0.55, s= 10)
            else:
                cax = ax03.scatter(x, y, c='w', edgecolor='k', alpha = 0.35, s= 10)
                cset = ax03.contour(xx, yy, f, colors='k', levels = 15, alpha = 0.85)
                ax02.clabel(cset, inline=1, fontsize=10)        
                # cax = ax03.scatter(x[idx_Tend], y[idx_Tend], c=masked_Tend,  cmap='bwr', alpha = 0.35, s= 10)
                cax = ax03.scatter(x[idx_Tend], y[idx_Tend], c=masked_Tend,  cmap='tab20c', alpha = 0.85, s= 10)
                print("MASKED T_inf: min beta=", np.argmin(x[idx_Tend]), "max beta=", np.argmax(x[idx_Tend]))
                print("MASKED T_inf: min gamma=", np.argmin(y[idx_Tend]), "max gamma=", np.argmax(y[idx_Tend]))
            
            
            ax03.grid(True, alpha=0.3)        
            ax03.set_xlabel(r"$\beta$", fontsize=20)
            ax03.set_ylabel(r"$\gamma^{-1}$", fontsize=20)
            for tick in ax03.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
            for tick in ax03.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)     
            if do_95 == 1:
                ax03.set_title(r'$95\% T_{end}$ Values')
            elif do_95 == 0:
                ax03.set_title(r'$68\% T_{end}$ Values')
            elif do_95 == -1:            
                ax03.set_title(r'$\mathcal{R}_0$ Slice $T_{end}$ Values')
            fig0.colorbar(cax, ax=ax03, shrink=0.9)

            # Global figure adjustments            
            fig0.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
            fig0.set_size_inches(29/2, 8/2, forward=True)

            if kwargs['store_plots']:
                if plot_regress_lines:
                    plt.savefig(kwargs['file_extension'] + "_CriticalPointsContours_regress.png", bbox_inches='tight')
                else: 
                    plt.savefig(kwargs['file_extension'] + "_CriticalPointsContours.png", bbox_inches='tight')
