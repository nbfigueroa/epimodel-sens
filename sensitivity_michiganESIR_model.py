from epimodels.sir import *
from epimodels.esir import *
from epimodels.utils import *
import xlsxwriter
import os


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


N = 1375987036
#time_points = []
#values = [] #As per the Mich report
#Initial cases as per March 4th

# time_points = [11, 15, 21, 71]
# values = [0.8,0.6,0.2, 1]
I0 = 25
R0 = 3
<<<<<<< HEAD
days = 750
#days = 90
=======
days = 712
# days = 90
>>>>>>> 31a5316a9ff06f7c0364db2fda6fa1b1bd8c6ca7

#R0 = 2.2, gamma = 1/7 no error

def run_without_error(r0 = 2.2, inf_period = 7, show_plot = True, verbose = True):
    if verbose:
        print(f'Scenario: r0 = {r0}, 1/gamma = {inf_period}')
    kwargs = {}
    kwargs['r0'] = r0
    kwargs['inf_period'] = inf_period
    kwargs['I0'] = I0
    kwargs['R0'] = R0
    
    
    
    model = eSIR(N, time_points, values, **kwargs)
    
    S,I,R = model.project(days)
    if show_plot:
        #plt.figure()
        plt.plot(I)
    
    #Compute IPeak
    I_peak = np.max(I)
    if verbose:
        print('The maximum active infections were: {I_peak:.3f}')
    
    #Compute the final infected population
    T_inf = N - S[-1]
    T_inf_prop = 1 - S[-1]/N
    if verbose: 
        print(f'The total number of people eventually infected: {T_inf:.3f}')
    
    #Compute the critical date (peak infections)
    r_t = []
    for (i,s) in enumerate(S):
        scaling = model.determine_scaling(i)
        val = scaling*model.args['r0']*s/model.N
        r_t.append(val)
    r_t = np.array(r_t)
    critical_day = np.argmin(np.abs(r_t-1))
    if verbose:
        print('The infections will peak after: ', critical_day , ' days\n\n')
    
    return (I_peak, T_inf, critical_day, (S,I,R))

def run_with_samples(r0_samples, inf_period_samples):
    
    I_peaks = {}
    I_peaks = []
    T_infs = {}
    T_infs = []
    critical_days = {}
    critical_days = []
    S = []
    I = []
    R = []
    
    for (r0, inf_period) in zip(r0_samples, inf_period_samples):
        I_peak, T_inf, critical_day, SIR = run_without_error(r0 = r0, inf_period = inf_period, show_plot = False, verbose = False)
        I_peaks.append(I_peak)
        T_infs.append(T_inf)
        critical_days.append(critical_day)
        S.append(SIR[0])
        I.append(SIR[1])
        R.append(SIR[2])
    
    m = np.mean(I_peaks)
    sd = np.std(I_peaks)
    up = m + 2*sd
    print('The peak active infections are: ', m, ' \[', up,'   ', m-2*sd, '\]')
    
    m = np.mean(T_infs)
    sd = np.std(T_infs)
    print('The total number of people eventually infected: ', m, ' [',m+2*sd, ', ',m-2*sd, ']')
    
    m = np.mean(critical_days)
    sd = np.std(critical_days)
    print('The infections will peak after: ', m, ' [',m+2*sd, ', ',m-2*sd, ']', ' days\n\n')
    
    return I_peaks, T_infs, critical_days, (S,I,R)
    

def mc_vary_r0 (r0_mean = 2.2, err = 0.05, inf_period = 7, n_samples = 5000):
    
    r0_sd = r0_mean*err
    print(f'Scenario: r0 sampled from N({r0_mean}, {r0_sd}, 1/gamma: 7, err {err}')
    
    
    #r0_samples = np.random.normal(r0_mean, r0_sd, n_samples)
    
    #inf_period_samples = inf_period*np.ones(r0_samples.shape)
    
    I_peak, T_inf, critical_days, SIR = run_without_error(r0_mean, inf_period, show_plot = True, verbose = True)
    I_peak_up, T_inf_up, critical_days_up, SIR = run_without_error(r0_mean+2*r0_sd, inf_period, show_plot = True, verbose = True)
    I_peak_d, T_inf_d, critical_days_d, SIR = run_without_error(r0_mean-2*r0_sd, inf_period, show_plot = True, verbose = True)
    
    print('The peak active infections are: ', I_peak, ' [', I_peak_d,',', I_peak_up, ']')
    print('The total number of people eventually infected: ', T_inf, ' [',T_inf_d, ', ',T_inf_up, ']')
    print('The infections will peak after: ', critical_days, ' [',critical_days_d, ', ',critical_days_up, ']', ' days\n\n')
    
    return I_peak, T_inf, critical_day, SIR


def mc_vary_inf (r0 = 2.2, err = 0.05, inf_period = 7, n_samples = 5000):
    
    inf_sd = inf_period*err
    print(f'Scenario: r0 sampled from N({r0}, {inf_sd}, 1/gamma: 7, err {err}')
    
    
    I_peak, T_inf, critical_days, SIR = run_without_error(r0, inf_period, show_plot = False, verbose = False)
    I_peak_up, T_inf_up, critical_days_up, SIR = run_without_error(r0, inf_period+2*inf_sd, show_plot = False, verbose = False)
    I_peak_d, T_inf_d, critical_days_d, SIR = run_without_error(r0, inf_period-2*inf_sd, show_plot = False, verbose = False)
    
    print('The peak active infections are: ', I_peak, ' [', I_peak_up,',', I_peak_d, ']')
    print('The total number of people eventually infected: ', T_inf, ' [',T_inf_up, ', ',T_inf_d, ']')
    print('The infections will peak after: ', critical_days, ' [',critical_days_up, ', ',critical_days_d, ']', ' days\n\n')
    
    return I_peak, T_inf, critical_day, SIR

def mc_vary_both(r0, inf_period, err):
    inf_sd = inf_period*err
    r0_sd = r0*err
    print(f'Scenario: r0 sampled from N({r0}, {inf_sd}, 1/gamma: 7, err {err}')
    
    
    I_peak, T_inf, critical_days, SIR = run_without_error(r0, inf_period, show_plot = False, verbose = False)
    I_peak_up, T_inf_up, critical_days_up, SIR = run_without_error(r0+2*r0_sd, inf_period+2*inf_sd, show_plot = False, verbose = False)
    I_peak_d, T_inf_d, critical_days_d, SIR = run_without_error(r0-2*r0_sd, inf_period-2*inf_sd, show_plot = False, verbose = False)
    
    print('The peak active infections are: ', I_peak, ' [', I_peak_up,',', I_peak_d, ']')
    print('The total number of people eventually infected: ', T_inf, ' [',T_inf_up, ', ',T_inf_d, ']')
    print('The infections will peak after: ', critical_days, ' [',critical_days_up, ', ',critical_days_d, ']', ' days\n\n')
    return I_peak, T_inf, critical_day, SIR

def run_without_error_beta(beta, inf_period, time_points, values, show_plot = True, verbose = True):
    
    r0 = beta*inf_period
    
    if verbose:
        print(f'Scenario: beta = {beta}, 1/gamma = {inf_period} r0 = {r0}')
        
    kwargs = {}
    kwargs['r0'] = r0
    kwargs['inf_period'] = inf_period
    kwargs['I0'] = I0
    kwargs['R0'] = R0
    
    
    
    model = eSIR(N, time_points, values, **kwargs)
    
    S,I,R = model.project(days)
    if show_plot:
        #plt.figure()
        plt.plot(I)
    
    #Compute IPeak
    I_peak = np.max(I)
    if verbose:
        print(f'The maximum active infections were: {I_peak:.3e}')
    
    #Compute the final infected population
    T_inf = N - S[-1]
    T_inf_prop = 1 - S[-1]/N
    if verbose: 
        print(f'The total number of people eventually infected: {T_inf:.3e}')
    
    #Compute the critical date (peak infections)
    r_t = []
    for (i,s) in enumerate(S):
        scaling = model.determine_scaling(i)
        val = scaling*model.args['r0']*s/model.N
        r_t.append(val)
    r_t = np.array(r_t)
    critical_day = np.argmin(np.abs(r_t-1))
    if verbose:
        print('The infections will peak after: ', critical_day , ' days\n\n')

    
    return (I_peak, T_inf, critical_day, (S,I,R))

def run_final(beta, inf_period, time_points, values, query_days = [42, 70, 90, 120], low_threshold = 10):
    
    I_peak, T_inf, critical_day, y = run_without_error_beta(beta, inf_period, time_points, values, show_plot=False, verbose=False)
    
    I = y[1]
    T = N - y[0]
    
    I_t = []
    T_t = []
    
    
    for d in query_days:
        I_t.append(I[d])
        T_t.append(T[d])
    
    t_lows = np.argwhere(I < low_threshold)
    t_low = np.min(t_lows)
    return I_t, T_t, I_peak, T_inf, critical_day, t_low, y
    
            
if __name__ == '__main__':

    ###########################################
    ######### Simulation parameters ###########
    ###########################################    
    Scenarios = [0,1,2]
    t = [[],[11,15,21,42],[11,15,21,70]]
    v = [[],[0.8,0.6,0.2,1],[0.8,0.6,0.2,1]]
    query_days = [42, 70, 90, 120]
    err = 0.1
    r0 = 1.78
    inf_period = 1/0.119
    beta = r0/inf_period
    beta_up = beta*(1+err)
    beta_down = beta*(1-err)
        
    betas = [beta, beta_up, beta_down]
    #inf_period = 1/0.119
    gamma_inv = 1/inf_period


    ########################################################
    ######### Plotting and storing parameters ##############
    ########################################################
    x_axis_offset       = 250
    y_axis_offset       = 0.0000000003
    store_plots         = 1 
    plot_all            = 1
    plot_peaks          = 1
    show_S              = 1
    show_R              = 0
    show_T              = 1    
    plot_superimposed   = 1
    show_analytic_limit = 0
    scale_offset        = 0.01 
    title_scenario = [' [Scenario 0: No Intervention]', ' [Scenario 1: Short Lockdown]', ' [Scenario 2: Long Lockdown]']    
    
    for scenario, time_points, values in zip(Scenarios, t, v):    
        ######## Record predictions ########
        S_samples       = np.empty([3, days+1])
        I_samples       = np.empty([3, days+1])
        R_samples       = np.empty([3, days+1])
        file_extensions  = ["./results/Michigan_Scenario{scenario:d}".format(scenario=scenario), 
                            "./results/Michigan_Scenario{scenario:d}_beta{error:d}error_plus".format(scenario=scenario, error=int(err*100)),
                            "./results/Michigan_Scenario{scenario:d}_beta{error:d}error_minus".format(scenario=scenario, error=int(err*100))]

        filename = os.path.join('results',f'Michigan_Scenario{scenario}_days.xlsx')
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()
        
        header = ['beta', 'gamma', 'R_0',]
        TI = [l  for d in query_days for l in [f'I_{d}', f'T_{d}']]
        header.extend(TI)
        header.extend(['t_low', 't_c', 'I_peak', 'T_inf'])
        
        for i in range(len(header)):
            worksheet.write(0,i, header[i])
        
        #run sims for all beta values
        for (p,beta) in enumerate(betas):
            j=p+1
            r0 = beta*inf_period
            worksheet.write(j, 0, beta)
            worksheet.write(j, 1, 1/inf_period)
            worksheet.write(j, 2, r0)
            cell = 2+1
            I_t, T_t, I_peak, T_inf, critical_day, t_low, y = run_final(beta, inf_period, time_points, values)
            
            for (i,d) in enumerate(query_days):
                worksheet.write(j, cell, I_t[i])
                cell=cell+1
                worksheet.write(j, cell, T_t[i])
                cell = cell+1
            
            worksheet.write(j, cell, t_low)
            cell+=1
            worksheet.write(j, cell, critical_day)
            cell+=1
            worksheet.write(j, cell, I_peak)
            cell+=1
            worksheet.write(j, cell, T_inf)

            #####################################################################
            ######## Plots Simulation with point estimates of parameters ########
            #####################################################################
            S =  y[0]
            I =  y[1]
            R =  y[2]
            T = I + R
            t = np.arange(0,days+1,1) 

            txt_title     = r"COVID-19 Michigan SIR Model Dynamic" + title_scenario[scenario]
            # txt_title     = r"COVID-19 Michigan ESIR Model Dynamics [Scenario {scenario:d}] ($R_0^e$={R0:1.3f}, $\beta_e$={beta:1.4f}, 1/$\gamma$={gamma:1.1f})"
            SIRparams     = scenario, float(r0), beta, gamma_inv, N
            SIRvariables  = S, I, R, T, t
            stor_plots_ii = 0
            Plotoptions   = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset
            plotSIR_evolution(txt_title, SIRparams, SIRvariables, Plotoptions, store_plots, file_extensions[p])
    
            if plot_superimposed:
                # Storing run in matrix for post-processing
                S_samples[p,:] = S
                I_samples[p,:] = I
                R_samples[p,:] = R
    
        
        workbook.close()
        
        #####################################################################
        ######## Plots Simulation with point estimates of parameters ########
        #####################################################################
        if plot_superimposed:
            show_S        = 0
            show_T        = 1
            plot_peaks    = 1
<<<<<<< HEAD
            beta_error    = err
            Plotoptions   = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, beta_error
            text_error    = r"$\beta \pm %1.2f \beta $"%beta_error
            plotSIR_evolutionErrors(txt_title, SIRparams, S_samples, I_samples, R_samples, Plotoptions, text_error, store_plots, file_extensions[0])
        
        plt.show()
        
    
    
    
    
    # I_peak = {}
    # T_inf = {}
    # critical_day = {}
    # SIR = {}
    # 
    '''
    key = (2.2,7,0)
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error()
    
    key = (2.5, 7, 0)
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error(r0 = key[0], inf_period = key[1])
    
    key = (2.8, 7, 0)
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error(r0 = key[0], inf_period = key[1])
    '''
    '''
    key = (2.5, 4, 0)
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error(r0 = key[0], inf_period = key[1])
    
    key = (2.5, 10, 0)
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error(r0 = key[0], inf_period = key[1])
    
    '''
    '''
    for err in[0.05, 0.1,0.15,0.2]:
        key = (2.2,7, err)
        I_peak[key], T_inf[key], critical_day[key], SIR[key] = mc_vary_r0(r0_mean = key[0], err = err, inf_period = key[1])
    '''
    
    '''
    for err in[0.05, 0.1,0.15,0.2]:
        key = (2.8,7, err)
        I_peak[key], T_inf[key], critical_day[key], SIR[key] = mc_vary_r0(r0_mean = key[0], err = err, inf_period = key[1])
    '''
    '''
    for err in[0.05, 0.1,0.15,0.2]:
        key = (2.5,10, err)
        I_peak[key], T_inf[key], critical_day[key], SIR[key] = mc_vary_inf(r0 = key[0], err = err, inf_period = key[1])
    
    '''
    '''
    for err in[0.05, 0.1,0.15,0.2]:
        key = (2.5,7, err)
        I_peak[key], T_inf[key], critical_day[key], SIR[key] = mc_vary_both(r0 = key[0], err = err, inf_period = key[1])
    '''
    '''
    key = (2.2,7,0)
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error()
    
    time_points= [11, 15, 21, 42]
    values = [0.8,0.6,0.2, 1]
    I_peak[key], T_inf[key], critical_day[key], SIR[key] = run_without_error()  
    '''
    
    #Running with extended lockdown conditions
    