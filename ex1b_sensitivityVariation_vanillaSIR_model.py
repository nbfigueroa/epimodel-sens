import numpy as np
from   scipy.optimize import fsolve
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import rc

# For custom classes and functions
from epimodels.sir    import *
from epimodels.utils  import *
from epimodels.plots  import *
from epimodels.sims   import *


def run_SIR(**kwargs):
    '''
        Run a single simulation of SIR dynamics
    '''
    N          = kwargs['N']
    days       = kwargs['days']
    beta       = kwargs['beta']
    r0         = kwargs['r0']
    gamma_inv  = kwargs['gamma_inv']
    gamma      = 1.0 / gamma_inv

    print('*****   SIMULATING SIR MODEL DYNAMICS *****')    
    print('*****         Hyper-parameters        *****')
    print('N=',N,'days=', days, 'r0=',r0, 'gamma_inv (days) = ',gamma_inv)
    print('*****         Model-parameters        *****')
    print('beta=',beta,'gamma=',gamma)

    model_kwargs = {}
    model_kwargs['r0']         = r0
    model_kwargs['inf_period'] = gamma_inv
    model_kwargs['I0']         = kwargs['I0']
    model_kwargs['R0']         = kwargs['R0']

    model   = SIR(N,**model_kwargs)
    S,I,R   = model.project(days,'ode_int')
    T       = I + R    

    return S,I,R,T


def storeVariationResults(beta_samples, gamma_inv_samples, I_samples, R_samples, **kwargs):
    
    T_samples  = I_samples+ R_samples
    R0_samples = np.array(beta_samples) * np.array(gamma_inv_samples)
    tc         = np.argmax(I_samples, axis=1)
    
    # Store stats
    worksheet        = kwargs['worksheet']
    row_num          = kwargs['row_num']

    tests, days = T_samples.shape
    tc_samples = []; Ipeak_samples = []; Tend_samples = [];
    for test in range(tests):      
        tc_samples.append(tc[test])
        Ipeak_samples.append(I_samples[test,tc[test]])
        Tend_samples.append(T_samples[test,days-1])

    worksheet.write_row(row_num, 0,  beta_samples)
    worksheet.write_row(row_num, 3,  gamma_inv_samples)
    worksheet.write_row(row_num, 6,  R0_samples)    
    worksheet.write_row(row_num, 9,  tc_samples)
    worksheet.write_row(row_num, 12, Ipeak_samples)
    worksheet.write_row(row_num, 15, Tend_samples)



def run_SIR_wErrors(beta_samples, gamma_inv_samples, text_error, **kwargs):
    '''
        Run multiple SIR simulations (means +/- errors)
    '''
    scenario = kwargs['scenario']

    ######## Record predictions ########
    S_samples       = np.empty([3, kwargs['days']])
    I_samples       = np.empty([3, kwargs['days']])
    R_samples       = np.empty([3, kwargs['days']])

    ############################################################
    ######## Simulate Single Vanilla SIR Model Dynamics ########
    ############################################################
    for ii in range(len(beta_samples)):
        kwargs['beta']      = beta_samples[ii]
        kwargs['gamma_inv'] = gamma_inv_samples[ii]
        kwargs['r0']        = beta_samples[ii]*gamma_inv_samples[ii]
        S,I,R,T             = run_SIR(**kwargs)

        print('*********   Results    *********')    
        tc, t_I100, t_I500, t_I100, t_I10 = getCriticalPointsAfterPeak(I)
        T_tc  = T[tc]
        print('Total Cases @ Peak = ', T_tc,'by day=', tc)
        Peak_infected     = I[-1]
        print('Infected @ tc = ', Peak_infected)
        total_cases     = T[-1]
        print('Total Cases @ t(end) = ', total_cases)

        # Storing run in matrix for post-processing
        S_samples[ii,:] = S
        I_samples[ii,:] = I
        R_samples[ii,:] = R


    storeVariationResults(beta_samples, gamma_inv_samples, I_samples, R_samples, **kwargs)

    ##############################################################
    ######## Plots Simulation Variables with Error Bounds ########
    ##############################################################
    x_axis_offset       = round(kwargs['days']*0.4)
    y_axis_offset       = 0.0000000003 
    plot_all            = 1; plot_peaks = 1; show_S = 0; show_T = 1; show_R = 0; show_analytic_limit = 0; scale_offset = 0.01 
    Plotoptions         = plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, scale_offset, scenario
    plotSIR_evolutionErrors_new(S_samples, I_samples, R_samples, Plotoptions, text_error, **kwargs)    


def main():    

    ####################################################################
    ######## Choose Initial and Model Parameters for Simulation ########
    ####################################################################
    '''Simulation options defined in sims.py
        sim_num = 1  --> India case study
        sim_num = 2  --> Mexico case study
        sim_num = 3  --> US case study
        sim_num = 4  --> Yucatan case study
        sim_num = 5  --> Primer case study
    '''
    sim_num    = 5; scenario   = 0
    sim_kwargs             = loadSimulationParams(sim_num, scenario, plot_data = 0)
    # Need to get rid of this variable here/..
    sim_kwargs['scenario'] = scenario
    basefilename           = sim_kwargs['file_extension']
    workbook, worksheet    = createResultsfile(basefilename, 'errorVary', test_type='varying')

    ## For variation on these parameters
    beta       = sim_kwargs['beta']
    gamma_inv  = sim_kwargs['gamma_inv']

    # Variables for +/- errors on beta
    error_perc        = 10
    err               = error_perc/100
    
    ########### Test 1: Vary beta, fix gamma ############
    text_error                    = r"$\beta \pm %1.2f \beta $"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVaryBeta"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 1
    beta_samples                  = [beta, beta*(1+err), beta*(1-err)]
    gamma_inv_samples             = [gamma_inv, gamma_inv, gamma_inv]
    run_SIR_wErrors(beta_samples, gamma_inv_samples, text_error, **sim_kwargs)
    plt.show()

    ########### Test 2: fix beta, vary gamma ############
    text_error                    = r"$\gamma^{-1} \pm %1.2f \gamma^{-1} $"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVaryGamma"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 2
    beta_samples                  = [beta, beta, beta]
    gamma_inv_samples             = [gamma_inv, gamma_inv*(1+err), gamma_inv*(1-err)]
    run_SIR_wErrors(beta_samples, gamma_inv_samples, text_error, **sim_kwargs)
    plt.show()

    ########### Test 3: fix beta, vary gamma ############
    text_error                    =  r"$\beta \pm %1.2f \beta $"%err + "\n" +  r"$\gamma^{-1} \pm %1.2f \gamma^{-1} $"%err
    sim_kwargs['file_extension']  = basefilename + "_errorsVaryBetaGamma"
    sim_kwargs['worksheet']       = worksheet
    sim_kwargs['row_num']         = 3
    beta_samples      = [beta, beta*(1+err), beta*(1-err)]
    gamma_inv_samples = [gamma_inv, gamma_inv*(1+err), gamma_inv*(1-err)]
    run_SIR_wErrors(beta_samples, gamma_inv_samples, text_error, **sim_kwargs)
    plt.show()
    
    workbook.close()

if __name__ == '__main__':
    main()

### TODO!!!: ADD THESE PLOTS -- R_t and r_I
# #############################################################
# ######## Dependence of R0 on Final Epidemic Behavior ########
# #############################################################

# # Final epidemic size (analytic)
# r0_vals     = np.linspace(1,5,100) 
# init_guess  = 0.0001
# Sinf_N      =   []
# Sinf_S0     =   []
# for ii in range(len(r0_vals)):
#     r0_test = r0_vals[ii]
#     Sinf_N.append(fsolve(epi_size, init_guess))     
#     Sinf_S0.append(1 - Sinf_N[ii])

# r0_test      = r0
# covid_SinfN  = fsolve(epi_size, init_guess)
# covid_SinfS0 = 1 - covid_SinfN
# print('Covid r0 = ', r0_test, 'Covid Sinf/S0 = ', covid_SinfN[0], 'Covid Sinf/S0 = ', covid_SinfS0[0]) 

# # Plots
# fig0, ax0 = plt.subplots()
# ax0.plot(r0_vals, Sinf_S0, 'r', lw=2, label='Susceptible')
# ax0.set_ylabel('$S_{\infty}/S_{0}$ (percentage of population infected)', fontsize=12)
# ax0.set_xlabel('$R_0$', fontsize=12)

# # Current estimate of Covid R0
# plt.title('Final Size of Epidemic Dependence on $R_0$ estimate',fontsize=15)
# ax0.plot(r0_test, covid_SinfS0, 'ko', markersize=5, lw=2)

# # Plot mean
# txt = 'Covid R0({r0:3.3f})'
# ax0.text(r0_test - 0.45, covid_SinfS0 + 0.05,txt.format(r0=r0_test), fontsize=10)
# plt.plot([r0]*10,np.linspace(0,covid_SinfS0,10), color='black')
# txt = "{Sinf:3.3f} Infected"
# ax0.text(1.1, covid_SinfS0 - 0.025,txt.format(Sinf=covid_SinfS0[0]), fontsize=8)
# plt.plot(np.linspace(1,[r0],10), [covid_SinfS0]*10, color='black')

# ax0.text(4, 0.75, r"${\cal R}_0 \equiv \frac{ \beta } {\gamma}$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
# ax0.text(4, 0.65, r"$\beta \sim {\cal N}(\mu_{\beta},(\sigma_{\beta}^2)$", fontsize=15, bbox=dict(facecolor='red', alpha=0.15))
# ax0.text(4, 0.45, r"$\epsilon_{\beta}=$%.3f"%beta_error, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.15))
# ax0.text(4, 0.35, r"$\sigma_{\beta}=\mu_{\beta}\epsilon_{\beta}=$%.3f"%beta_sigma, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.15))

# # Sample R0 values from Gaussian distribution for Covid estimate
# # Variables for R0 distribution
# r0_mean    = gamma_inv*beta_mean
# r0_sigma   = np.sqrt(pow(gamma_inv,2)*pow(beta_sigma,2))
# r0_gauss   = stats.norm.pdf(r0_vals,r0_mean,r0_sigma)
# # r0_gauss   = gamma_inv*stats.norm.pdf(r0_vals/gamma_inv,beta_mean,beta_sigma)

# r0_samples = gamma_inv*np.random.normal(beta_mean, beta_sigma, 100)

# # Just for sanity check, actually the mean is the same only beta_std will be different
# beta_mean  = np.mean(r0_samples/gamma_inv)
# beta_std   = np.std(r0_samples/gamma_inv)
# beta_plus  = beta_mean + 2*beta_std
# beta_minus = beta_mean - 2*beta_std

# r0_plus    = r0_samples[np.argmax(r0_samples)]
# r0_minus   = r0_samples[np.argmin(r0_samples)]
# print(r0_plus, r0_minus)
# r0_plus    = gamma_inv*beta_plus
# r0_minus   = gamma_inv*beta_minus
# print('R0 =[',r0_minus, r0_mean, r0_plus)


# covid_SinfS0_samples  =   []
# for ii in range(len(r0_samples)):
#     r0_test = r0_samples[ii]
#     covid_SinfS0_samples.append(1 - fsolve(epi_size, init_guess))

# plt.plot(r0_samples, covid_SinfS0_samples, 'ko', markersize=5, lw=2, alpha=0.5)

# # Plot +std
# r0_test          = r0_plus
# covid_SinfS0_plus = 1 - fsolve(epi_size, init_guess)
# ax0.text(r0_test + 0.02, covid_SinfS0_plus - 0.15,r'$R_0+2\sigma_{R_0}$', fontsize=10)
# # txt = '$\sigma_{R_0}$=' + str(r0_sigma)
# # ax0.text(r0_test + 0.02, covid_SinfS0_plus - 0.5,txt, fontsize=10)
# plt.plot([r0_test]*10,np.linspace(0,covid_SinfS0_plus,10), color='black', alpha=0.5)
# txt = "{Sinf:3.3f} Infected"
# ax0.text(1.1, covid_SinfS0_plus + 0.01,txt.format(Sinf=covid_SinfS0_plus[0]), fontsize=8)
# plt.plot(np.linspace(1,[r0_test],10), [covid_SinfS0_plus]*10, color='black', alpha=0.5)


# # Plot -std
# r0_test          = r0_minus
# covid_SinfS0_plus = 1 - fsolve(epi_size, init_guess)
# ax0.text(r0_test - 0.45, covid_SinfS0_plus - 0.25,r'$R_0-2\sigma_{R_0}$', fontsize=10)
# plt.plot([r0_test]*10,np.linspace(0,covid_SinfS0_plus,10), color='black', alpha=0.5)
# txt = "{Sinf:3.3f} Infected"
# ax0.text(1.1, covid_SinfS0_plus - 0.02,txt.format(Sinf=covid_SinfS0_plus[0]), fontsize=8)
# plt.plot(np.linspace(1,[r0_test],10), [covid_SinfS0_plus]*10, color='black', alpha=0.5)

# ax0.xaxis.set_tick_params(length=0)
# ax0.yaxis.set_tick_params(length=0)

# # Plot Gaussian and sampled from R0
# ax01       = ax0.twinx()   # instantiate a second axes that shares the same x-axis
# plt.plot(r0_vals,4*r0_gauss/r0_mean, color='black', alpha=0.25)
# plt.plot(r0_samples, [0.15]*len(r0_samples), 'ko', markersize=5, lw=2, alpha=0.5)
# plt.ylim(0.1,5)
# ax01.xaxis.set_tick_params(length=0)
# ax01.yaxis.set_tick_params(length=0)
# fig0.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
# fig0.set_size_inches(18.5/2, 12.5/2, forward=True)

# plt.savefig('./results/vanillaSIR_finalSize_%i_beta_%i.png'%(sim_num,error_perc), bbox_inches='tight')
# plt.savefig('./results/vanillaSIR_finalSize_%i_beta_%i.pdf'%(sim_num,error_perc), bbox_inches='tight')

# #################################################################
# ######## Plots Simulation with reproductive/growth rates ########
# #################################################################
# # Analytic growth rate
# effective_Rt = r0_mean * (S/N)
# growth_rates = gamma * (effective_Rt - 1)

# effective_Rt_plus = r0_plus * (S_plus/N)
# growth_rates_plus = gamma * (effective_Rt_plus - 1)

# effective_Rt_minus = r0_minus * (S_minus/N)
# growth_rates_minus = gamma * (effective_Rt_minus - 1)

# ####### Plots for Growth Rates #######
# fig, (ax1, ax2) = plt.subplots(1,2)

# # Plot of Reproductive rate (number)
# ax1.plot(t, effective_Rt, 'k', lw=2, label='Rt (Effective Reproductive Rate)')
# ax1.text(t[0] + 0.02, effective_Rt[0] - 0.15,r'${\cal R}_t$', fontsize=10)
# ax1.plot(t, 1*np.ones(len(t)), 'r-')
# txt1 = "Critical (Rt={per:2.2f})"
# ax1.text(t[-1]-50, 1 + 0.01, txt1.format(per=1), fontsize=12, color='r')
# ax1.text(t[-1]-50,2.5, r"${\cal R}_t \equiv \left( \frac{S (t) }{N (t) } \right) {\cal R}_0$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))

# # sigma values
# ax1.plot(t, effective_Rt_plus, 'k', lw=2,  label='${\cal R}_t + 2\sigma$', alpha=0.5)
# ax1.text(t[0] + 0.02, effective_Rt_plus[0] - 0.15,r'${\cal R}_t+2\sigma$', fontsize=10)
# ax1.plot(t, effective_Rt_minus, 'k', lw=2, label='${\cal R}_t - 2\sigma$', alpha=0.5)
# ax1.text(t[0] - 0.02, effective_Rt_minus[0] - 0.15,r'${\cal R}_t-2\sigma$', fontsize=10)

# # Estimations of End of Epidemic
# effRT_diff     = effective_Rt - 1
# ids_less_1     = np.nonzero(effRT_diff < 0)
# effRT_crossing = ids_less_1[0][0]

# effRT_diff_plus     = effective_Rt_plus - 1
# ids_less_1_plus     = np.nonzero(effRT_diff_plus < 0)
# effRT_crossing_plus = ids_less_1_plus[0][0]

# effRT_diff_minus     = effective_Rt_minus - 1
# ids_less_1_minus     = np.nonzero(effRT_diff_minus < 0)
# effRT_crossing_minus = ids_less_1_minus[0][0]
# print('R_t= [', effRT_crossing_plus, effRT_crossing, effRT_crossing_minus,']')

# err_range = abs(effRT_crossing_plus-effRT_crossing_minus)
# txt1 = "Est. error [{err_range:2.2f} days]"
# ax1.text(t[-1]-50, 1 - 0.2, txt1.format(err_range=err_range), fontsize=12, color='r')

# ax1.plot(effRT_crossing, 1,'ro', markersize=12)
# ax1.text(effRT_crossing-10, 1-0.2,str(effRT_crossing), fontsize=10, color="r")
# ax1.plot(effRT_crossing_plus, 1,'ro', markersize=12)
# ax1.text(effRT_crossing_plus-10, 1-0.2,str(effRT_crossing_plus), fontsize=10, color="r")
# ax1.plot(effRT_crossing_minus, 1,'ro', markersize=12)
# ax1.text(effRT_crossing_minus-10, 1-0.2,str(effRT_crossing_minus), fontsize=10, color="r")


# ax1.set_ylabel('Rt (Effective Reproductive Rate)', fontsize=12)
# ax1.set_xlabel('Time[days]', fontsize=12)
# ax1.set_ylim(0,4)
# fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
# txt_title = "COVID-19 SIR Model Dynamics (N={N:2.0f},R0={R0:1.3f},1/gamma={gamma_inv:1.3f}, beta={beta:1.3f})"
# fig.suptitle(txt_title.format(N=N, R0=r0, gamma_inv = gamma_inv, beta= beta),fontsize=15)

# # Plot of temporal growth rate
# ax2.plot(t, growth_rates, 'k', lw=2, label='rI (temporal growth rate)')
# ax2.text(t[0] + 0.02, growth_rates[0] - 0.02,r'${r}_I(t)$', fontsize=10)

# ax2.plot(t, 0*np.ones(len(t)), 'r-')
# txt1 = r"Critical ($r_I$={per:2.2f})"
# ax2.text(t[-1]-50, 0 + 0.01, txt1.format(per=0), fontsize=12, color='r')
# ax2.text(t[-1]-50, 0.2, r"$r_I  \equiv \gamma \left[ {\cal R}_t - 1 \right]$", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))
# ax2.text(t[-1]-50, 0.1, r"$\frac{ dI}{dt} = r_I \, I $", fontsize=15, bbox=dict(facecolor='red', alpha=0.2))

# # sigma values
# ax2.plot(t, growth_rates_plus, 'k', lw=2,  label='$Rt + 2\sigma$', alpha=0.5)
# ax2.text(t[0] + 0.02, growth_rates_plus[0] - 0.02,r'${r}_I(t)+2\sigma$', fontsize=10)
# ax2.plot(t, growth_rates_minus, 'k', lw=2, label='$Rt - 2\sigma$', alpha=0.5)
# ax2.text(t[0] + 0.02, growth_rates_minus[0] - 0.02,r'${r}_I(t)-2\sigma$', fontsize=10)

# ax2.set_ylabel('rI (temporal growth rate)', fontsize=12)
# ax2.set_xlabel('Time[days]',fontsize=12)
# ax2.set_ylim(-0.2,0.5)

# # Estimations of End of Epidemic
# rI_diff     = growth_rates 
# ids_less_0  = np.nonzero(rI_diff < 0)
# rI_crossing = ids_less_1[0][0]

# rI_diff_plus     = growth_rates_plus 
# ids_less_1_plus  = np.nonzero(rI_diff_plus < 0)
# rI_crossing_plus = ids_less_1_plus[0][0]

# rI_diff_minus     = growth_rates_minus
# ids_less_1_minus  = np.nonzero(rI_diff_minus < 0)
# rI_crossing_minus = ids_less_1_minus[0][0]
# print('R_t= [', rI_crossing_plus, rI_crossing, rI_crossing_minus,']')

# err_range = abs(rI_crossing_plus-rI_crossing_minus)
# txt1 = "Est. error [{err_range:2.2f} days]"
# ax2.text(t[-1]-50, 0 - 0.04, txt1.format(err_range=err_range), fontsize=12, color='r')

# ax2.plot(rI_crossing, 0,'ro', markersize=12)
# ax2.text(rI_crossing-10, 0-0.04,str(rI_crossing), fontsize=10, color="r")
# ax2.plot(rI_crossing_plus, 0,'ro', markersize=12)
# ax2.text(rI_crossing_plus-15, 0-0.04,str(rI_crossing_plus), fontsize=10, color="r")
# ax2.plot(rI_crossing_minus, 0,'ro', markersize=12)
# ax2.text(rI_crossing_minus-10, 0-0.04,str(rI_crossing_minus), fontsize=10, color="r")
# fig.set_size_inches(27.5/2, 12.5/2, forward=True)

# plt.savefig('./results/vanillaSIR_growthRates_%i_beta_%i.png'%(sim_num,error_perc), bbox_inches='tight')
# plt.savefig('./results/vanillaSIR_growthRates_%i_beta_%i.pdf'%(sim_num,error_perc), bbox_inches='tight')

# plt.show()



