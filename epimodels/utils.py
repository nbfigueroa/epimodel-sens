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
<<<<<<< HEAD
=======
def plotIR_realizations(I_samples, R_samples, **kwargs):
    # Plot random realizations of Infected and Recovered Curves
    fig, (ax0,ax01) = plt.subplots(1,2, constrained_layout=True)
    N = len(I_samples)
    if N > 1000:
        N = 1000
    for ii in range(N):
        ax0.plot(I_samples[ii,:]/kwargs['N'])
    ax0.set_title(r"Realizations of $I(t=0)\rightarrow I(t=200)$", fontsize=20)
    ax0.grid(True, alpha=0.3)
    ax0.set_ylabel('Fraction of the Population', fontsize=20)
    ax0.set_xlabel('[Time/days]', fontsize=20)
    for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax0.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 

    for ii in range(1000):
        ax01.plot(R_samples[ii,:]/kwargs['N'])
    ax01.set_title(r"Realizations of $R(t=0)\rightarrow I(t=200)$", fontsize=20)
    ax01.grid(True, alpha=0.3)
    ax01.set_ylabel('Fraction of the Population', fontsize=20)
    ax01.set_xlabel('[Time/days]', fontsize=20)
    for tick in ax01.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax01.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(20/2, 8/2, forward=True)
    if kwargs['store_plots']:
        plt.savefig(kwargs['file_extension'] + "_IRrealizations.png", bbox_inches='tight')


    
    


def plotInfected_evolution(Ivariables, Plotoptions, number_scaling = 'million', **kwargs):
    # Unpacking variables
    I, t = Ivariables
    T_limit, plot_peaks, x_axis_offset, y_axis_offset = Plotoptions 


    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    filename       = kwargs['file_extension']        
    store_plots    = kwargs['store_plots']
    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, T_limit, kwargs['x_tick_step'])

    infected_data = np.array([]); infected_estimated_data = np.array([]); check_points = np.array([])
    if 'Infected' in kwargs:
        infected_data            = kwargs['Infected']
    if 'Infected_est' in kwargs:
        infected_estimated_data  = kwargs['Infected_est']
    if 'check_points' in kwargs:
        check_points             = kwargs['check_points']

    I_plot = I[0:T_limit]
    t_plot = t[0:T_limit]
    
    if number_scaling == 'million':
        scale      = 1000000
    elif number_scaling == '100k':
        scale      = 100000
    elif number_scaling == '10k':
        scale      = 10000
    elif number_scaling == 'k':
        scale      = 1000
    elif number_scaling == 'none':
        scale          = 1 
        number_scaling = ""

    if plot_peaks:
        tc    =  np.argmax(I_plot)
        I_tc  = I_plot[tc]

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)    
    fig.suptitle(figure_title,fontsize=20)

    #####   Variable evolution in linear scale    #####
    ax1.plot(t_plot, I_plot/N, 'r',   lw=2,   label='Infected')
    if infected_data.size > 0:
        t_infected = infected_data.size
        ax1.plot(np.arange(0,t_infected,1), infected_data/N, 'bo',  markersize=5,  alpha= 0.5, label='Confirmed Infected')

    if infected_estimated_data.size > 0:
        t_estimated_infected = infected_estimated_data.size
        ax1.plot(np.arange(0,t_estimated_infected,1), infected_estimated_data/N, 'mo',  markersize=5,  alpha= 0.5, label='Sentinal Infected')

    if plot_peaks:
        # Plot peak points
        ax1.plot(tc, I_tc/N,'ro', markersize=8)        
        txt_title = r"Peak Infected: {I_tc:2.4f} {number_scaling} by day {peak_days:10.0f} " 
        ax1.text(0.5*tc, 0.9*I_tc/N, txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling,  peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    # Making things beautiful
    ax1.set_ylabel('Fraction of Population', fontsize=20)    
    legend = ax1.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)    

    ax1.grid(True, color='k', alpha=0.2, linewidth = 0.25)        

    if check_points.size > 0:
        print('Check-points given!')
        # Plot peak points
        ax1.plot(check_points, I_plot[check_points]/N,'ko', alpha = 0.5)     


    #####   Variable evolution in log scale    #####
    ax2.plot(t_plot, I_plot/N, 'r',   lw=2,   label='Infected')
    if infected_data.size > 0:
        ax2.plot(np.arange(0,t_infected,1), infected_data/N, 'bo',  markersize=5,  alpha= 0.5, label='Confirmed Infected')
    if infected_estimated_data.size > 0:        
        ax2.plot(np.arange(0,t_estimated_infected,1), infected_estimated_data/N, 'mo',  markersize=5,  alpha= 0.5, label='Sentinal Infected')

    if plot_peaks:
        # Plot peak points
        ax2.plot(tc, I_tc/N,'ro', markersize=8)    
        
        min_peaks, _ = find_peaks(-np.log(I_plot), distance=2)
        max_peaks, _ = find_peaks(np.log(I_plot), distance=2)
        
        txt_title = r"Infected: {I_tc:2.4f} {number_scaling} by day {peak_days:10.0f} " 
        if max_peaks.size > 0:
            # this way the x-axis corresponds to the index of x
            ax2.plot(max_peaks[0], I_plot[max_peaks[0]]/N, 'ro',  markersize=5,  alpha= 0.5)            
            ax2.text(max_peaks[0]*0.75, 1.5*I_plot[max_peaks[0]]/N, txt_title.format(I_tc=I_plot[max_peaks[0]]/scale, number_scaling=number_scaling,  peak_days= max_peaks[0]), fontsize=15, color="r",  bbox=dict(facecolor='white', alpha=0.75))
        if min_peaks.size > 0:
            ax2.plot(min_peaks[0], I_plot[min_peaks[0]]/N, 'ro',  markersize=5,  alpha= 0.5)
            ax2.text(min_peaks[0]*0.75, 0.5*I_plot[min_peaks[0]]/N, txt_title.format(I_tc=I_plot[min_peaks[0]]/scale, number_scaling=number_scaling,  peak_days= min_peaks[0]), fontsize=15, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if check_points.size > 0:
        txt_title = r"Infected: {I_tc:2.4f} {number_scaling} by day {peak_days:10.0f} " 
        ax2.plot(check_points, I_plot[check_points]/N,'k+', markersize=8, alpha = 0.5, label='Check-points')    
        for ii in range(check_points.size):
            ax2.text(check_points[ii]+2, I_plot[check_points[ii]]/N, txt_title.format(I_tc=I_plot[check_points[ii]]/scale, number_scaling=number_scaling,  peak_days= check_points[ii]), fontsize=11.5, color="k", bbox=dict(facecolor='white', alpha=0.75))
    
    plt.yscale("log")

    ax2.set_xlabel('Time /days', fontsize=20)
    ax2.set_ylabel('Fraction of Population', fontsize=20)
    if 'x_tick_names' in kwargs:
        plt.xticks(x_tick_numbers, x_tick_names)
    legend = ax2.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax2.spines[spine].set_visible(True)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)     
    ax2.grid(True, color='k', alpha=0.2, linewidth = 0.25)        


    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 20.5/2, forward=True)
    
    if store_plots:
        plt.savefig(filename + "_infected_" + str(T_limit) + ".png", bbox_inches='tight')
        # plt.savefig(filename + "_infected_" + str(T_limit) + ".pdf", bbox_inches='tight')


>>>>>>> aebaf7fc8012248158a54f5cb384a92c226c13e4
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

<<<<<<< HEAD
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
=======

############################################################################################################
#############                                PLOTS FOR SIR MODEL                               #############
############################################################################################################
# def plotSIR_evolution(figure_title, SIRvariables, N, Plotoptions, filename, number_scaling = 'million', x_tick_labels = []):
def plotSIR_evolution(SIRvariables, Plotoptions, number_scaling = 'million',**kwargs):
    
    # Unpacking variables
    S, I, R, T, t = SIRvariables
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset = Plotoptions 
    
    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    filename       = kwargs['file_extension']

    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, len(T), kwargs['x_tick_step'])

    
    store_plots    = kwargs['store_plots']
    # Check if x_tick_labels is given
    # x_tick_numbers, x_tick_names = x_tick_labels
    
    if number_scaling == 'million':
        scale      = 1000000
    elif number_scaling == '100k':
        scale      = 100000
    elif number_scaling == '10k':
        scale      = 10000
    elif number_scaling == 'k':
        scale      = 1000
    elif number_scaling == 'none':
        scale     = 1 
        number_scaling = ""
    
    if plot_peaks:
        tc, t_I100, t_I500, t_I100, t_I10 = getCriticalPointsAfterPeak(I)
        tc    =  np.argmax(I)
        I_tc  = I[tc]
        T_tc  = T[tc]
    
    total_cases     = T[-1]

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()    
    # fig.suptitle(txt_title.format(scenario=scenario, R0=float(r0), beta= beta, gamma = 1/gamma_inv),fontsize=20)
    fig.suptitle(figure_title,fontsize=20)

    # Variable evolution    
    ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')

    if plot_all:        
        show_S = 1
        show_R = 1
        show_T = 1
        # Plot Final Epidemic Size
        if show_analytic_limit:
            ax1.plot(t, One_SinfN*np.ones(len(t)), 'm--')
            txt1 = "Analytic Epidemic Size: 1-S(inf)/N={per:2.2f} percentage (analytic)"
            ax1.text(t[-1]-200, One_SinfN + 0.02, txt1.format(per=One_SinfN[0]), fontsize=20, color='m')
    
    if show_T:
        ax1.plot(t, T/N, 'm', lw=2,   label='Total Cases')
        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.3f} million total cases as $t(end)$"
        ax1.text(t[-1] - x_axis_offset, (total_cases/N) + y_axis_offset, txt1.format(per=total_cases/scale), fontsize=20, color='r')    

    if show_S:
        ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
    
    if show_R:
        ax1.plot(t, R/N, 'g--',  lw=1,  label='Recovered')

    if plot_peaks:
        # Plot peak points
        ax1.plot(tc, I_tc/N,'ro', markersize=8)        
        txt_title = r"Peak infected: {I_tc:2.4f} {number_scaling} by day {peak_days:10.0f} " 
        txt_title2 = r"Total Cases: {peak_total:2.4f} {number_scaling} by day {peak_days:10.0f} " 
        ax1.text(1.1*tc, I_tc/N, txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling,  peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if show_T:
            ax1.plot(tc, T_tc/N,'ro', markersize=8)
            ax1.text(1.1*tc, T_tc/N, txt_title2.format(peak_total=T_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

    # Making things beautiful
    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
    if 'x_tick_names' in kwargs:
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
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)
    
    if store_plots:
        plt.savefig(filename + ".png", bbox_inches='tight')
        # plt.savefig(filename + ".pdf", bbox_inches='tight')


def plotSIR_growth(Svariables, **kwargs):    
    # Unpacking Data and Plotting Options    
    S, t           = Svariables    
    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    r0             = kwargs['r0']
    gamma          = 1/kwargs['gamma_inv']
    filename       = kwargs['file_extension']
    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, len(S), kwargs['x_tick_step'])
    store_plots    = kwargs['store_plots']

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(figure_title,fontsize=20)    
    
    ########################################################
    #######       Plots for Reproductive Rates     #########
    ########################################################
    effective_Rt = r0 * (S/N)
    
    # Plot of Reproductive rate (number)
    ax1.plot(t, effective_Rt, 'k', lw=2)
    ax1.text(t[0] + 0.02, effective_Rt[0] - 0.15,r'${\cal R}_t(t)$', fontsize=20)
    ax1.plot(t, 1*np.ones(len(t)), 'r-')
    txt1 = "Critical (Rt={per:2.2f})"
    ax1.text(t[-1]-50, 1 + 0.01, txt1.format(per=1), fontsize=12, color='r')
    ax1.grid(True, color='k', alpha=0.2, linewidth = 0.25)        


    # Estimations of critical point of epidemic
    tcs_Reff  = np.nonzero(effective_Rt  < 1.0001)
    a = np.array(tcs_Reff)
    if a.size > 0:
        tc_Reff =  tcs_Reff[0][0]
        print('R_t <= 1 @ day=', tc_Reff, 'R_t=',effective_Rt[tc_Reff])
        print('Previous=',effective_Rt[tc_Reff-1])
        ax1.plot(tc_Reff, 1,'ro', markersize=12)
        ax1.text(tc_Reff*1.1,0.9,str(tc_Reff), fontsize=20, color="r")
    else: 
        tc_Reff = Nan

    # Making things beautiful
    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('$\mathcal{R}_t$ (Effective Reproductive Rate)', fontsize=20)
    if 'x_tick_names' in kwargs:
        ax1.set_xticks(x_tick_numbers)
        ax1.set_xticklabels(x_tick_names)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)         

    ##################################################
    #######       Plots for Growth Rates     #########
    ##################################################
    growth_rates = gamma * (effective_Rt - 1)
    ax2.plot(t, growth_rates, 'k', lw=2)
    ax2.text(t[0] + 0.02, growth_rates[0] - 0.02,r'${r}_I(t)$', fontsize=20)

    ax2.plot(t, 0*np.ones(len(t)), 'r-')
    txt1 = r"Critical ($r_I$={per:2.2f})"
    ax2.text(t[-1]-50, 0 + 0.01, txt1.format(per=0), fontsize=12, color='r')

    tcs_growth  = np.nonzero(growth_rates  < 0)
    a = np.array(tcs_growth)
    if a.size > 0:
        tc_growth =  tcs_growth[0][0]
        print('r_I <= 0 @ day=', tc_growth, 'r_I=', growth_rates[tc_growth])
        print('Previous=',growth_rates[tc_growth-1])
        ax2.plot(tc_growth, 0,'ro', markersize=12)
        ax2.text(tc_growth*1.1,-0.02,str(tc_growth), fontsize=20, color="r")
    else: 
        tc_growth = Nan
    

    # Making things beautiful
    ax2.set_ylabel('$r_I$(temporal growth rate)', fontsize=20)
    ax2.set_xlabel('Time[days]',fontsize=20)
    if 'x_tick_names' in kwargs:
        ax2.set_xticks(x_tick_numbers)
        ax2.set_xticklabels(x_tick_names)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)     

    ax2.grid(True, color='k', alpha=0.2, linewidth = 0.25)        

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(35.5/2, 14.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + "_growthRates.png", bbox_inches='tight')
        # plt.savefig(filename + ".pdf", bbox_inches='tight')
    
    return tc_Reff, effective_Rt[tc_Reff], tc_growth, growth_rates[tc_growth]


def plotSIR_finalEpidemicR0(**kwargs):

    r0             = kwargs['r0']
    filename       = kwargs['file_extension']
    store_plots    = kwargs['store_plots']

    # Equation to estimate final epidemic size (infected)
    def epi_size(x):
        return np.log(x) + r0_test*(1-x)

    # Final epidemic size (analytic)
    r0_vals     = np.linspace(1,5,100) 
    init_guess  = 0.0001
    Sinf_N      =   []
    Sinf_S0     =   []
    for ii in range(len(r0_vals)):
        r0_test = r0_vals[ii]
        Sinf_N.append(fsolve(epi_size, init_guess))     
        Sinf_S0.append(1 - Sinf_N[ii])

    r0_test      = r0
    covid_SinfN  = fsolve(epi_size, init_guess)
    covid_SinfS0 = 1 - covid_SinfN
    print('Covid r0 = ', r0_test, 'Covid Sinf/S0 = ', covid_SinfN[0], 'Covid Sinf/S0 = ', covid_SinfS0[0]) 

    # Plots
    fig0, ax0 = plt.subplots()
    ax0.plot(r0_vals, Sinf_S0, 'r', lw=2, label='Susceptible')
    ax0.set_ylabel('1 - $S_{\infty}/S_{0}$ (Fraction of Population Infected)', fontsize=20)
    ax0.set_xlabel('$\mathcal{R}_0$', fontsize=20)

    for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax0.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)     

    # Current estimate of Covid R0
    plt.title('Final Size of Epidemic Dependence on $\mathcal{R}_0$ estimate',fontsize=20)
    ax0.plot(r0_test, covid_SinfS0, 'ko', markersize=5, lw=2)

    # Plot mean
    txt = 'Covid $R_0$({r0:3.3f})'
    ax0.text(r0_test - 0.45, covid_SinfS0 + 0.05,txt.format(r0=r0_test), fontsize=20)
    plt.plot([r0]*10,np.linspace(0,covid_SinfS0,10), color='black')
    txt = "{Sinf:3.3f} Infected"
    ax0.text(1.1, covid_SinfS0*0.95,txt.format(Sinf=covid_SinfS0[0]), fontsize=15)
    plt.plot(np.linspace(1,[r0],10), [covid_SinfS0]*10, color='black')

    ax0.text(4, 0.75, r"${\cal R}_0 \equiv \frac{ \beta } {\gamma}$", fontsize=20, bbox=dict(facecolor='red', alpha=0.15))
    ax0.grid(True, color='k', alpha=0.2, linewidth = 0.25)        
    fig0.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig0.set_size_inches(25.5/2, 12.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + "_finalEpidemic.png", bbox_inches='tight')
        # plt.savefig(filename + ".pdf", bbox_inches='tight')


def plotSIR_evolutionErrors_new(S_variables, I_variables, R_variables, Plotoptions, text_error, **kwargs):
    scale = 1000000        

    S       = S_variables[0,:]
    S_plus  = S_variables[1,:]
    S_minus = S_variables[2,:]

    I       = I_variables[0,:]
    I_plus  = I_variables[1,:]
    I_minus = I_variables[2,:]

    R       = R_variables[0,:]
    R_plus  = R_variables[1,:]
    R_minus = R_variables[2,:]


    T = I + R
    T_minus = I_minus+ R_minus
    T_plus = I_plus+ R_plus

    Tf = len(T_plus)
    t = np.arange(0, Tf, 1)

    # Unpack
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset, scenario  = Plotoptions 

    # Unpacking Simulation and Plotting Options    
    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    r0             = kwargs['r0']
    gamma          = 1/kwargs['gamma_inv']
    filename       = kwargs['file_extension']
    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, len(S), kwargs['x_tick_step'])
    store_plots    = kwargs['store_plots']

    # Plot the data of three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()
    fig.suptitle(figure_title,fontsize=25)    

    # Variable evolution
    if show_S:
        ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.25)
        ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
        ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.25)

    ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, I/N, 'r', lw=2,   label='Infected Cases')
    ax1.plot(t, I_minus/N, 'r--', lw=2, alpha=0.25)
    # scenario = 2
    if show_T:
        ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
        ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
        ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)

        total_cases     = T[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (total_cases/N), txt1.format(per=total_cases/scale), fontsize=18, color='m')

        total_cases     = T_minus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, 0.98*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=18, color='m')

        total_cases     = T_plus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=18, color='m')
        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)

        ax1.text(1, 0.5, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    else:
        if Tf == 90:
            if scenario == 1:
                ax1.text(0.2*Tf, 0.0012, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
            else:
                ax1.text(0.5*Tf, 0.0000007, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    # Estimated Final epidemic size (analytic) not-dependent on simulation

    # Equation to estimate final epidemic size (infected)
    def epi_size(x):        
        return np.log(x) + r0_test*(1-x)

    init_guess   = 0.0001
    r0_test      = float(r0)
    SinfN  = fsolve(epi_size, init_guess)
    One_SinfN = 1 - SinfN
    print('*****   Final Epidemic Size    *****')
    print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

    print('*****   Results    *****')
    tc =  np.argmax(I)
    I_tc     = I[tc]
    print('Peak Instant. Infected = ', I_tc,'by day=', tc)

    T_tc  = T[tc]
    print('Total Cases when Peak = ', T_tc,'by day=', tc)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    I_tc_plus_idx =  np.argmax(I_plus)
    I_tc_plus     = I_plus[I_tc_plus_idx]
    print('Peak Instant. Infected - Error= ', I_tc_plus,'by day=', I_tc_plus_idx)

    I_tc_minus_idx =  np.argmax(I_minus)
    I_tc_minus     = I_minus[I_tc_minus_idx]
    print('Peak Instant. Infected + Error= ', I_tc_minus,'by day=', I_tc_minus_idx)

    do_plus = 1; do_minus = 1
    if abs(tc-I_tc_plus_idx) < 3:
        do_plus = 0
    if abs(tc-I_tc_minus_idx) < 3:
        do_minus = 0

    if plot_peaks:
        # Plot peak points
        ax1.plot(tc, I_tc/N,'ro', markersize=8)
        if do_plus:
            # Plot peak points
            ax1.plot(I_tc_plus_idx, I_tc_plus/N,'ro', markersize=8)
        if do_minus:
            # Plot peak points
            ax1.plot(I_tc_minus_idx, I_tc_minus/N,'ro', markersize=8)

        if Tf == 90:
            if scenario == 2:
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(tc+ 5, I_tc/N , txt_title.format(I_tc=I_tc, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 30, 0.9*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+ 5,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            else:
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(tc- 40, I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 20, I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx -30,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        else:
            # Adjust automatically
            txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+2, (1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_plus:        
                txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx-25, (1 + 10*scale_offset)*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_minus:
                txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+2, (1 - 10*scale_offset)*I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(tc, T_tc/N,'mo', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="m", bbox=dict(facecolor='white', alpha=0.75))


    ax1.set_xlabel('Time /days', fontsize=30)
    ax1.set_ylabel('Fraction of Population', fontsize=30)
    # ax1.yaxis.set_tick_params(length=0)
    # ax1.xaxis.set_tick_params(length=0)
    if 'x_tick_names' in kwargs:
        ax1.set_xticks(x_tick_numbers)
        ax1.set_xticklabels(x_tick_names)
    
    legend = ax1.legend(fontsize=20, loc='center right')
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 
    
    plt.grid(b=True, which='major', c='w', lw=2, ls='-')
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + ".png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')


def plotSIR_evolutionStochastic(S_variables, I_variables, R_variables, Plotoptions, text_error, **kwargs):
    scale = 1000000        

    S       = S_variables[0,:]
    S_plus  = S_variables[1,:]
    S_minus = S_variables[2,:]
    S_std   = S_variables[3,:]

    I       = I_variables[0,:]
    I_plus  = I_variables[1,:]
    I_minus = I_variables[2,:]
    I_std   = I_variables[3,:]

    R       = R_variables[0,:]
    R_plus  = R_variables[1,:]
    R_minus = R_variables[2,:]
    R_std   = R_variables[3,:]

    T = I + R
    T_minus = I_minus+ R_minus
    T_plus  = I_plus + R_plus
    T_std   = I_std  + R_std

    Tf = len(T_plus)
    t = np.arange(0, Tf, 1)

    # Unpack
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset  = Plotoptions 

    # Unpacking Simulation and Plotting Options    
    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    r0             = kwargs['r0']
    gamma          = 1/kwargs['gamma_inv']
    filename       = kwargs['file_extension']
    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, len(S), kwargs['x_tick_step'])
    store_plots    = kwargs['store_plots']

    # Plot the data of three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()
    fig.suptitle(figure_title,fontsize=25)    

    # Variable evolution
    if show_S:
        ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.35)
        ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
        ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.35)
        # ax1.fill_between(t,(S - S_std)/N,(S + S_std)/N, color="#348ABD", alpha=0.15)
        ax1.fill_between(t,(S - S_std)/N,(S + S_std)/N, color='k', alpha=0.15)

    ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, I/N, 'r', lw=2,   label='Infected Cases')
    ax1.plot(t, I_minus/N, 'r--', lw=2, alpha=0.25)
    ax1.fill_between(t,(I - I_std)/N,(I + I_std)/N, color='r', alpha=0.10)

    scenario = 2
    if show_T:
        ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
        ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
        ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)
        ax1.fill_between(t,(T - T_std)/N,(T + T_std)/N, color='m', alpha=0.10)

        total_cases     = T[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (total_cases/N), txt1.format(per=total_cases/scale), fontsize=12, color='m')

        total_cases     = T_minus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, 0.98*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=12, color='m')

        total_cases     = T_plus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=12, color='m')
        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)

        ax1.text(1, 0.5, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    else:
        if Tf == 90:
            if scenario == 1:
                ax1.text(0.2*Tf, 0.0012, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
            else:
                ax1.text(0.5*Tf, 0.0000007, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    

    print('*****   Results    *****')
    tc =  np.argmax(I)
    I_tc     = I[tc]
    print('Peak Instant. Infected = ', I_tc,'by day=', tc)

    T_tc  = T[tc]
    print('Total Cases when Peak = ', T_tc,'by day=', tc)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    I_tc_plus_idx =  np.argmax(I_plus)
    I_tc_plus     = I_plus[I_tc_plus_idx]
    print('Peak Instant. Infected - Error= ', I_tc_plus,'by day=', I_tc_plus_idx)

    I_tc_minus_idx =  np.argmax(I_minus)
    I_tc_minus     = I_minus[I_tc_minus_idx]
    print('Peak Instant. Infected + Error= ', I_tc_minus,'by day=', I_tc_minus_idx)

    do_plus = 1; do_minus = 1
    if abs(tc-I_tc_plus_idx) < 3:
        do_plus = 0
    if abs(tc-I_tc_minus_idx) < 3:
        do_minus = 0

    if plot_peaks:
        # Plot peak points
        ax1.plot(tc, I_tc/N,'ro', markersize=8)
        if do_plus:
            # Plot peak points
            ax1.plot(I_tc_plus_idx, I_tc_plus/N,'ro', markersize=8)
        if do_minus:
            # Plot peak points
            ax1.plot(I_tc_minus_idx, I_tc_minus/N,'ro', markersize=8)

        if Tf == 90:
            if scenario == 2:
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(tc+ 5, I_tc/N , txt_title.format(I_tc=I_tc, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 30, 0.9*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+ 5,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            else:
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(tc- 40, I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 20, I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx -30,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        else:
            # Adjust automatically
            txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+2, (1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_plus:        
                txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx-25, (1 + 10*scale_offset)*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_minus:
                txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+2, (1 - 10*scale_offset)*I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(tc, T_tc/N,'mo', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="m", bbox=dict(facecolor='white', alpha=0.75))


    ax1.set_xlabel('Time /days', fontsize=30)
    ax1.set_ylabel('Fraction of Population', fontsize=30)
    if 'x_tick_names' in kwargs:
        ax1.set_xticks(x_tick_numbers)
        ax1.set_xticklabels(x_tick_names)
    
    legend = ax1.legend(fontsize=20, loc='center right')
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 
    
    # plt.grid(b=True, which='major', c='w', lw=2, ls='-')
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + ".png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')
        

def plotSIR_evolutionStochastic2(S_samples, I_samples, R_samples, Plotoptions, text_error, CI = 0.95, **kwargs):
    scale = 1000000        
    
    upper_quantile = 1 - (1-CI)/2
    lower_quantile = (1-CI)/2

    # S       = S_variables[0,:]
    # S_plus  = S_variables[1,:]
    # S_minus = S_variables[2,:]
    # S_std   = S_variables[3,:]

    # I       = I_variables[0,:]
    # I_plus  = I_variables[1,:]
    # I_minus = I_variables[2,:]
    # I_std   = I_variables[3,:]

    # R       = R_variables[0,:]
    # R_plus  = R_variables[1,:]
    # R_minus = R_variables[2,:]
    # R_std   = R_variables[3,:]

    # T = I + R
    # T_minus = I_minus+ R_minus
    # T_plus  = I_plus + R_plus
    # T_std   = I_std  + R_std
    
    S           = np.mean(S_samples, axis=0)
    S_plus      = np.quantile(S_samples, upper_quantile, axis = 0)
    S_minus     = np.quantile(S_samples, lower_quantile, axis = 0)
    
    I           = np.mean(I_samples, axis = 0)
    I_plus      = np.quantile(I_samples, upper_quantile, axis = 0)
    I_minus     = np.quantile(I_samples, lower_quantile, axis = 0)
    
    R           = np.mean(R_samples, axis = 0)
    R_plus      = np.quantile(R_samples, upper_quantile, axis = 0)
    R_minus     = np.quantile(R_samples, lower_quantile, axis = 0)
    
    T_samples   = I_samples + R_samples
    T           = I + R
    T_plus      = np.quantile(T_samples, upper_quantile, axis = 0)
    T_minus     = np.quantile(T_samples, lower_quantile, axis = 0)

    Tf = len(T_plus)
    t = np.arange(0, Tf, 1)

    # Unpack
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset  = Plotoptions 

    # Unpacking Simulation and Plotting Options    
    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    r0             = kwargs['r0']
    gamma          = 1/kwargs['gamma_inv']
    filename       = kwargs['file_extension']
    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, len(S), kwargs['x_tick_step'])
    store_plots    = kwargs['store_plots']

    # Plot the data of three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()
    fig.suptitle(figure_title,fontsize=25)    

    # Variable evolution
    if show_S:
        ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.35)
        ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
        ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.35)
        # ax1.fill_between(t,(S - S_std)/N,(S + S_std)/N, color="#348ABD", alpha=0.15)
        ax1.fill_between(t, S_plus/N, S_minus/N, color='k', alpha=0.15)

    ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, I/N, 'r', lw=2,   label='Infected Cases')
    ax1.plot(t, I_minus/N, 'r--', lw=2, alpha=0.25)
    ax1.fill_between(t,I_plus/N,I_minus/N, color='r', alpha=0.10)

    scenario = 2
    if show_T:
        ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
        ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
        ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)
        ax1.fill_between(t,T_plus/N,T_minus/N, color='m', alpha=0.10)

        total_cases     = T[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (total_cases/N), txt1.format(per=total_cases/scale), fontsize=12, color='m')

        total_cases     = T_minus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, 0.98*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=12, color='m')

        total_cases     = T_plus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=12, color='m')
        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)

        ax1.text(1, 0.5, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    else:
        if Tf == 90:
            if scenario == 1:
                ax1.text(0.2*Tf, 0.0012, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
            else:
                ax1.text(0.5*Tf, 0.0000007, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    

    print('*****   Results    *****')
    tc =  np.argmax(I)
    I_tc     = I[tc]
    print('Peak Instant. Infected = ', I_tc,'by day=', tc)

    T_tc  = T[tc]
    print('Total Cases when Peak = ', T_tc,'by day=', tc)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    I_tc_plus_idx =  np.argmax(I_plus)
    I_tc_plus     = I_plus[I_tc_plus_idx]
    print('Peak Instant. Infected - Error= ', I_tc_plus,'by day=', I_tc_plus_idx)

    I_tc_minus_idx =  np.argmax(I_minus)
    I_tc_minus     = I_minus[I_tc_minus_idx]
    print('Peak Instant. Infected + Error= ', I_tc_minus,'by day=', I_tc_minus_idx)

    do_plus = 1; do_minus = 1
    if abs(tc-I_tc_plus_idx) < 3:
        do_plus = 0
    if abs(tc-I_tc_minus_idx) < 3:
        do_minus = 0

    if plot_peaks:
        # Plot peak points
        ax1.plot(tc, I_tc/N,'ro', markersize=8)
        if do_plus:
            # Plot peak points
            ax1.plot(I_tc_plus_idx, I_tc_plus/N,'ro', markersize=8)
        if do_minus:
            # Plot peak points
            ax1.plot(I_tc_minus_idx, I_tc_minus/N,'ro', markersize=8)

        if Tf == 90:
            if scenario == 2:
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(tc+ 5, I_tc/N , txt_title.format(I_tc=I_tc, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 30, 0.9*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+ 5,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            else:
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(tc- 40, I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 20, I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx -30,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        else:
            # Adjust automatically
            txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+2, (1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_plus:        
                txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx-25, (1 + 10*scale_offset)*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_minus:
                txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+2, (1 - 10*scale_offset)*I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(tc, T_tc/N,'mo', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="m", bbox=dict(facecolor='white', alpha=0.75))


    ax1.set_xlabel('Time /days', fontsize=30)
    ax1.set_ylabel('Fraction of Population', fontsize=30)
    if 'x_tick_names' in kwargs:
        ax1.set_xticks(x_tick_numbers)
        ax1.set_xticklabels(x_tick_names)
    
    legend = ax1.legend(fontsize=20, loc='center right')
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 
    
    # plt.grid(b=True, which='major', c='w', lw=2, ls='-')
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + ".png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')
        





# Deprecate this soon:
def plotSIR_evolutionErrors(txt_title, SIRparams, S_variables, I_variables, R_variables, Plotoptions, text_error, store_plots, filename):
    scale = 1000000        

    # Unpack
    scenario, r0, beta, gamma_inv, N = SIRparams
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset = Plotoptions 

    S       = S_variables[0,:]
    S_plus  = S_variables[1,:]
    S_minus = S_variables[2,:]

    I       = I_variables[0,:]
    I_plus  = I_variables[1,:]
    I_minus = I_variables[2,:]

    R       = R_variables[0,:]
    R_plus  = R_variables[1,:]
    R_minus = R_variables[2,:]


    T = I + R
    T_minus = I_minus+ R_minus
    T_plus = I_plus+ R_plus

    Tf = len(T_plus)
    t = np.arange(0, Tf, 1)

    # Plot the data of three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()
    fig.suptitle(txt_title.format(scenario=scenario, R0=float(r0), beta= beta, gamma = 1/gamma_inv),fontsize=20)    

    # Variable evolution
    if show_S:
        ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.25)
        ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
        ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.25)

    ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, I/N, 'r', lw=2,   label='Infected Cases')
    ax1.plot(t, I_minus/N, 'r--', lw=2, alpha=0.25)
    scenario = 2
    if show_T:
        ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
        ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
        ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)

        total_cases     = T[-1]
        print('Total Cases when growth linear = ', total_cases)
        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.2f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (total_cases/N), txt1.format(per=total_cases/scale), fontsize=20, color='r')

        total_cases     = T_minus[-1]
        print('Total Cases when growth linear = ', total_cases)
        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.2f} million total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=20, color='r')

        total_cases     = T_plus[-1]
        print('Total Cases when growth linear = ', total_cases)
        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.2f} million total cases as $t(end)$"

        ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=20, color='r')
        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
        ax1.text(1, 0.5, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    else:
        if Tf == 90:
            if scenario == 1:
                ax1.text(0.2*Tf, 0.0012, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
            else:
                ax1.text(0.5*Tf, 0.0000007, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
    # Estimated Final epidemic size (analytic) not-dependent on simulation

    # Equation to estimate final epidemic size (infected)
    def epi_size(x):        
        return np.log(x) + r0_test*(1-x)

    init_guess   = 0.0001
    r0_test      = float(r0)
    SinfN  = fsolve(epi_size, init_guess)
    One_SinfN = 1 - SinfN
    print('*****   Final Epidemic Size    *****')
    print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

    print('*****   Results    *****')
    tc =  np.argmax(I)
    I_tc     = I[tc]
    print('Peak Instant. Infected = ', I_tc,'by day=', tc)

    T_tc  = T[tc]
    print('Total Cases when Peak = ', T_tc,'by day=', tc)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    I_tc_plus_idx =  np.argmax(I_plus)
    I_tc_plus     = I_plus[I_tc_plus_idx]
    print('Peak Instant. Infected - Error= ', I_tc_plus,'by day=', I_tc_plus_idx)

    I_tc_minus_idx =  np.argmax(I_minus)
    I_tc_minus     = I_minus[I_tc_minus_idx]
    print('Peak Instant. Infected + Error= ', I_tc_minus,'by day=', I_tc_minus_idx)

    if plot_peaks:
        # Plot peak points
        ax1.plot(tc, I_tc/N,'ro', markersize=8)
        # Plot peak points
        ax1.plot(I_tc_plus_idx, I_tc_plus/N,'ro', markersize=8)
        # Plot peak points
        ax1.plot(I_tc_minus_idx, I_tc_minus/N,'ro', markersize=8)
        

        if Tf == 90:
            if scenario == 2:
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(tc+ 5, I_tc/N , txt_title.format(I_tc=I_tc, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 30, 0.9*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+ 5,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            else:
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(tc- 40, I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 20, I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f}mill. by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx -30,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        else:
            # Adjust automatically
            txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+2, (1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(I_tc_plus_idx-25, (1 + 10*scale_offset)*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(I_tc_minus_idx+2, (1 - 10*scale_offset)*I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(tc, T_tc/N,'mo', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))


    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    
    legend = ax1.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 

    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + ".png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')





###########################################################################################################
#############                              PLOTS FOR SEIR MODEL                               #############
###########################################################################################################
##..
##..
##..
##..
##..
##..
##..
##..
##..
##..
##..
##..
##..



############################################################################################################
#############                              PLOTS FOR SEIQR MODEL                               #############
############################################################################################################
def plotSEIQR_evolution(txt_title, SEIQRparams, SEIQRvariables, Plotoptions, store_plots, filename):
    scenario, r0, beta, gamma_inv, sigma_inv, tau_q_inv, q, N = SEIQRparams
    S, E, I, Q, Re, D, t = SEIQRvariables
    R     = Re + D + Q
    T     = I + R 
    Inf   = I + Q
    All_I = I + Q + E 


    plot_all, show_S, show_E, show_Q, show_R, show_D, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset = Plotoptions

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    scale = 1000000
    total_cases     = T[-1]

    if plot_peaks:
        peak_All_I_idx =  np.argmax(All_I)
        peak_All_I     = All_I[peak_All_I_idx]

        I_tce_idx =  np.argmax(Inf)
        I_tce     = Inf[I_tce_idx]

        tc =  np.argmax(I)
        I_tc     = I[tc]

        if show_Q:
            peak_Q_idx =  np.argmax(Q)
            peak_Q     = Q[peak_Q_idx]

        if show_E:
            peak_E_idx =  np.argmax(E)
            peak_E     = E[peak_E_idx]

        if plot_all:
            T_tc  = T[tc]


    fig, ax1 = plt.subplots()
    fig.suptitle(txt_title.format(R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, tau_q_inv = tau_q_inv, q=q),fontsize=20)

    # Variable evolution
    ax1.plot(t, All_I/N, 'g', lw=2,   label='Active (E+I+Q)')
    ax1.plot(t, I/N, 'r',     lw=2,   label='Infected')
    ax1.plot(t, D/N, 'b--',   lw=1,   label='Dead')
    ax1.plot(t, Inf/N, 'r--', lw=2,   label='Infectuos (I+Q)')

    if show_Q:
        ax1.plot(t, Q/N, 'c',     lw=2,   label='Quarantined')  
    if show_E:
        ax1.plot(t, E/N, 'm',   lw=2, label='Exposed')
    if plot_all:
        ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.2f} million total cases as t(end)."
        # ax1.text(t[0], (total_cases/N) - 0.05, txt1.format(per=total_cases/scale), fontsize=20, color='r')
        ax1.text(t[-1]-x_axis_offset, (total_cases/N) - 0.05, txt1.format(per=total_cases/scale), fontsize=20, color='r')
        if show_S:
            ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
        if show_R:
            ax1.plot(t, Re/N, 'g--',  lw=1,  label='Recovered')
            ax1.plot(t, R/N, 'g',  lw=2,  label='Recovered+Dead+Quarantined')

        # Plot Final Epidemic Size
        if show_analytic_limit:        
            ax1.plot(t, One_SinfN*np.ones(len(t)), 'm--')
            txt1 = "Final Epidemic size (no intervention): 1-S(inf)/N={per:2.2f} percentage (analytic)"
            ax1.text(t[-1]-200, One_SinfN + 0.02, txt1.format(per=One_SinfN[0]), fontsize=20, color='m')

    # Plot peak points
    ax1.plot(tc, I_tc/N,'ro', markersize=8)
    txt_title = r"Peak infected: {I_tc:5.5f}million by day {peak_days:10.0f}" 
    txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f}" 
    ax1.text(tc+10, I_tc/N, txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if plot_all:
        ax1.plot(tc, T_tc/N,'ro', markersize=8)
        ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))
        
    if plot_peaks:        
        # Plot peak points
        ax1.plot(I_tce_idx, I_tce/N,'ro', markersize=8)
        txt_title3 = r"Peak Infectuous (I+Q): {I_tce:5.5f}million by day {peake_days:10.0f}" 
        ax1.text(I_tce_idx+10, I_tce/N, txt_title3.format(I_tce=I_tce/scale, peake_days= I_tce_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.95))
        
        if show_Q:
            ax1.plot(peak_Q_idx, peak_Q/N,'ro', markersize=8)
            txt_title3 = r"Peak Quarantined: {peak_Q:5.5f}million by day {peake_days:10.0f}" 
            ax1.text(peak_Q_idx+10, peak_Q/N, txt_title3.format(peak_Q=peak_Q/scale, peake_days= peak_Q_idx), fontsize=20, color="r",bbox=dict(facecolor='white', alpha=0.75))
        
        if show_E:
            ax1.plot(peak_E_idx, peak_E/N,'ro', markersize=8)
            txt_title3 = r"Peak Exposed: {peak_E:5.5f}million by day {peake_days:10.0f}" 
            ax1.text(peak_E_idx+10, peak_E/N, txt_title3.format(peak_E=peak_E/scale, peake_days= peak_E_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

        ax1.plot(peak_All_I_idx, peak_All_I/N,'ro', markersize=8)
        txt_title3 = r"Peak Active: {peak_E:5.5f}million by day {peake_days:10.0f}" 
        ax1.text(peak_All_I_idx+10, peak_All_I/N, txt_title3.format(peak_E=peak_All_I/scale, peake_days= peak_All_I_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

    # Making things beautiful
    ax1.set_xlabel('Time /days', fontsize=40)
    ax1.set_ylabel('Fraction of Population', fontsize=40)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax1.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)


    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 


    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + '.png', bbox_inches='tight')
        # plt.savefig(filename + '.pdf', bbox_inches='tight')



def plotSEIQR_evolutionErrors(txt_title, SEIQRparams, S_variables, E_variables, I_variables,  Q_variables, 
    Re_variables, D_variables, Plotoptions, text_error, store_plots, filename):
    scale = 1000000    
    
    # Unpack
    scenario, r0, beta, gamma_inv, sigma_inv, tau_q_inv, q, N = SEIQRparams
    plot_all, show_S, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, beta_error = Plotoptions 

    S       = S_variables[0,:]
    S_plus  = S_variables[1,:]
    S_minus = S_variables[2,:]

    E       = E_variables[0,:]
    E_plus  = E_variables[1,:]
    E_minus = E_variables[2,:]

    I       = I_variables[0,:]
    I_plus  = I_variables[1,:]
    I_minus = I_variables[2,:]

    Q       = Q_variables[0,:]
    Q_plus  = Q_variables[1,:]
    Q_minus = Q_variables[2,:]

    Re       = Re_variables[0,:]
    Re_plus  = Re_variables[1,:]
    Re_minus = Re_variables[2,:]

    D        = D_variables[0,:]
    D_plus   = D_variables[1,:]
    D_minus  = D_variables[2,:]


    R     = Re + D + Q
    T     = I + R 
    Inf   = I + Q
    All_I = I + Q + E 

    T = I + R

    R_plus     = Re_plus + D_plus + Q_plus
    T_plus     = I_plus + R_plus 
    Inf_plus   = I_plus + Q_plus
    All_I_plus = I_plus + Q_plus + E_plus 
    T_plus     = I_plus+ R_plus

    R_minus     = Re_minus + D_minus + Q_minus
    T_minus     = I_minus + R_minus 
    Inf_minus   = I_minus + Q_minus
    All_I_minus = I_minus + Q_minus + E_minus
    T_minus     = I_minus+ R_minus
    

    Tf = len(T_plus)
    t = np.arange(0, Tf, 1)

    # Plot the data of three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()
    fig.suptitle(txt_title.format(R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, tau_q_inv = tau_q_inv, q=q),fontsize=20)
    ax1.text(Tf/2, 0.95, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))

    # Variable evolution
    ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.25)
    ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
    ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.25)

    ax1.plot(t, Inf_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, Inf/N, 'r', lw=2,   label='Infectuous (I+Q)')
    ax1.plot(t, Inf_minus/N, 'r--', lw=2, alpha=0.25)

    ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
    ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
    ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)
    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as $t(end)$"
    ax1.text(t[-1]-x_axis_offset, (total_cases/N), txt1.format(per=total_cases/scale), fontsize=20, color='r')

    total_cases     = T_minus[-1]
    print('Total Cases when growth linear = ', total_cases)
    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as $t(end)$"
    ax1.text(t[-1]-x_axis_offset, 0.95*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=20, color='r')

    total_cases     = T_plus[-1]
    print('Total Cases when growth linear = ', total_cases)
    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as $t(end)$"
    ax1.text(t[-1]-x_axis_offset, 1.05*(total_cases/N), txt1.format(per=total_cases/scale), fontsize=20, color='r')
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)

    # Estimated Final epidemic size (analytic) not-dependent on simulation

    # Equation to estimate final epidemic size (infected)
    def epi_size(x):        
        return np.log(x) + r0_test*(1-x)

    init_guess   = 0.0001
    r0_test      = float(r0)
    SinfN  = fsolve(epi_size, init_guess)
    One_SinfN = 1 - SinfN
    print('*****   Final Epidemic Size    *****')
    print('r0 = ', r0_test, '1 - Sinf/S0 = ', One_SinfN[0])    

    print('*****   Results    *****')
    tc =  np.argmax(Inf)
    I_tc     = Inf[tc]
    print('Peak Instant. Infected = ', I_tc,'by day=', tc)

    T_tc  = T[tc]
    print('Total Cases when Peak = ', T_tc,'by day=', tc)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    I_tc_plus_idx =  np.argmax(Inf_plus)
    I_tc_plus     = Inf_plus[I_tc_plus_idx]
    print('Peak Instant. Infected - Error= ', I_tc_plus,'by day=', I_tc_plus_idx)

    I_tc_minus_idx =  np.argmax(Inf_minus)
    I_tc_minus     = Inf_minus[I_tc_minus_idx]
    print('Peak Instant. Infected + Error= ', I_tc_minus,'by day=', I_tc_minus_idx)

    # Plot peak points
    ax1.plot(tc, I_tc/N,'ro', markersize=8)
    # Plot peak points
    ax1.plot(I_tc_plus_idx, I_tc_plus/N,'ro', markersize=8)
    # Plot peak points
    ax1.plot(I_tc_minus_idx, I_tc_minus/N,'ro', markersize=8)
    
    txt_title = r"Peak infectuous (I+Q): {I_tc:5.5f}million by day {peak_days:10.0f} " 
    ax1.text(tc+10, (1.1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
    if plot_peaks:
        txt_title = r"Peak infectuous (I+Q): {I_tc:5.5f}million by day {peak_days:10.0f} " 
        ax1.text(I_tc_plus_idx+5, (1.4)*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
        txt_title = r"Peak infectuous (I+Q): {I_tc:5.5f}million by day {peak_days:10.0f} " 
        ax1.text(I_tc_minus_idx+5, (0.6)*I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if plot_all == 1:
        ax1.plot(tc, T_tc/N,'ro', markersize=8)
        txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
        ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

    ####### OPTIONAL STUFF #######  
    if show_analytic_limit:
        ax1.plot(t, covid_SinfS0*np.ones(len(t)), 'm--')
        txt1 = "{per:2.2f} population infected"
        ax1.text(t[0], covid_SinfS0 - 0.05, txt1.format(per=covid_SinfS0[0]), fontsize=20, color='m')


    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax1.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20) 

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + "_all.png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')

>>>>>>> aebaf7fc8012248158a54f5cb384a92c226c13e4
