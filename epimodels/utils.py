import math
import numpy  as np
import pandas as pd 
from   scipy.optimize import fsolve
from   scipy.signal   import find_peaks

import matplotlib.pyplot as plt
from   matplotlib import rc
# For beautiful plots
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

###########################################################################################################
#############                          FUNCTIONS FOR DATA LOADING                             ############
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


########################################################################################################
#############                       Functions FOR ALL MODELS                               #############
########################################################################################################
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

def getCriticalPointsStats(I_samples, R_samples):

    # Compute Total Cases
    T_samples = I_samples + R_samples

    n_samples, n_days = I_samples.shape

    tc_samples       = np.empty([n_samples, 1])
    Ipeak_samples    = np.empty([n_samples, 1])
    Tend_samples     = np.empty([n_samples, 1])
    
    for ii in range(n_samples):
        tc, _             = find_peaks(I_samples[ii,:], distance=1)
        tc_samples[ii]    = tc
        Ipeak_samples[ii] = I_samples[ii,tc]
        Tend_samples[ii]  = T_samples[ii,n_days-1]
    
    return tc_samples, Ipeak_samples, Tend_samples


def computeStats(X, bound_type='CI'):
    '''
        Compute mean, median and confidence intervals/quantiles
    '''
    X_bar = np.mean(X, axis=0)    # mean of vector
    X_std = np.std(X, axis=0)     # std of vector
    X_med = np.median(X, axis=0)  # median of vector

    # Computing 95% Confidence Intervals
    if bound_type == 'CI':
        n     = len(X_bar) # number of obs
        z     = 1.96 # for a 95% CI
        X_lower = X_bar - (z * (X_std/math.sqrt(n)))
        X_upper = X_bar + (z * (X_std/math.sqrt(n)))

    if bound_type == 'Quantiles':    
        X_lower = np.quantile(X, 0.025, axis = 0)
        X_upper = np.quantile(X, 0.975, axis = 0)
           
    return X_bar, X_med, X_std, X_upper, X_lower

def gatherMCstats(S_samples, I_samples, R_samples, bound_type='CI'):    
    '''
        Gather stats from MC simulations  
    '''
    T_samples = I_samples + R_samples
    S_mean, S_med, S_std, S_upper, S_lower = computeStats(S_samples, bound_type)
    I_mean, I_med, I_std, I_upper, I_lower = computeStats(I_samples, bound_type)
    R_mean, R_med, R_std, R_upper, R_lower = computeStats(R_samples, bound_type)
    T_mean, T_med, T_std, T_upper, T_lower = computeStats(T_samples, bound_type)

    # Pack values for plotting and analysis
    S_stats          = np.vstack((S_mean, S_med, S_upper, S_lower))    
    I_stats          = np.vstack((I_mean, I_med, I_upper, I_lower))    
    R_stats          = np.vstack((R_mean, R_med, R_upper, R_lower))    
    T_stats          = np.vstack((T_mean, T_med, T_upper, T_lower))    
    return S_stats, I_stats, R_stats, T_stats
