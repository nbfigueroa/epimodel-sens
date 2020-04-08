from epimodels.sir import *
from epimodels.esir import *

N = 1375987036
time_points = []
values = [] #As per the Mich report
#Initial cases as per March 4th

# time_points = [11, 15, 21, 71]
# values = [0.8,0.6,0.2, 1]
I0 = 25
R0 = 3
days = 365

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

def run_without_error_beta(beta, inf_period, show_plot = True, verbose = True):
    
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

        



if __name__ == '__main__':
    
    I_peak = {}
    T_inf = {}
    critical_day = {}
    SIR = {}
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
    