import numpy as np
# import GPy

from   scipy.optimize    import fsolve
from   scipy.signal      import find_peaks
from   scipy             import stats
from   scipy.stats       import gamma as gamma_dist
from   scipy.stats       import kde
from scipy.interpolate   import UnivariateSpline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from   matplotlib import rc
# For beautiful plots
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)


import statsmodels.api       as sm
from statsmodels.formula.api import ols
import pandas as pd
from   epimodels.utils import *


############################################################################################################
#############                                PLOTS FOR ALL MODEL                               #############
############################################################################################################
def unpack_simulationOptions():
    # This function will unpack all plotting./simulation arguments
    return 1

def plotIT_realizations(I_samples, R_samples, **kwargs):

    # Compute random id's to plot realizations
    N     = len(I_samples)
    idx   = np.arange(0,N-1,1)
    idx_r = np.random.permutation(idx)

    if N < 1001:
        N_realizations = N-1
    else:
        N_realizations = 1000

    # Compute Total Cases
    T_samples = I_samples + R_samples

    # Plot random realizations of Infectedes and Total Cases Curves
    fig, (ax0,ax01) = plt.subplots(1,2, constrained_layout=True)

    for ii in range(N_realizations):
        ax0.plot(I_samples[idx_r[ii],:]/kwargs['N'])
    ax0.set_title(r"Realizations of $I(t=0)\rightarrow I(t=end)$", fontsize=20)
    ax0.grid(True, alpha=0.3)
    ax0.set_ylabel('Fraction of the Population', fontsize=20)
    ax0.set_xlabel('[Time/days]', fontsize=20)
    for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax0.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 

    for ii in range(N_realizations):
        ax01.plot(T_samples[idx_r[ii],:]/kwargs['N'])
    ax01.set_title(r"Realizations of $T(t=0)\rightarrow T(t=end)$", fontsize=20)
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
        plt.savefig(kwargs['file_extension'] + "_ITrealizations.png", bbox_inches='tight')


# TODO: Move these to utils.py
def fit1D_KDE(x):
    # Use scipy.stats class to fit the bandwith
    tc_kde = stats.gaussian_kde(x.T)    
    stdev = np.sqrt(tc_kde.covariance)[0, 0]

    # using statsmodels kde to compute quantiles
    tc_kde_sm = sm.nonparametric.KDEUnivariate(x)    
    tc_kde_sm.fit()
    bw = tc_kde_sm.bw
    tc_kde_sm.fit(bw=np.max(np.array([stdev,bw])))
    icdf_spl = UnivariateSpline(np.linspace(0, 1, num = tc_kde_sm.icdf.size), tc_kde_sm.icdf)    

    # fig00, ax00 = plt.subplots()    
    # ax00.plot(np.linspace(0, 1, num = tc_kde_sm.icdf.size),tc_kde_sm.icdf,'k', lw=1, label=r"icdf-kde")
    # ax00.plot(np.linspace(0, 1, num = tc_kde_sm.icdf.size),icdf_spl(np.linspace(0, 1, num = tc_kde_sm.icdf.size)),'b', lw=1, label=r"icdf-spl")
  
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
    R0_samples = beta_samples * gamma_inv_samples

    ############################################################################################################
    #######  Compute Descriptive Stats for each critical point distributions (t_c, I_peak, T_end and R0) #######
    ############################################################################################################

    ##### Stats on inputs to model #####
    beta_bar, beta_med, beta_std, beta_upper95, beta_lower95 = computeStats(beta_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, beta_upper68, beta_lower68 = computeStats(beta_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    beta_skew = stats.skew(beta_samples)    
    print('Mean beta=',beta_bar, ' Med beta=', beta_med, 'Skew beta=', beta_skew)

    gamma_inv_bar, gamma_inv_med, gamma_inv_std, gamma_inv_upper95, gamma_inv_lower95 = computeStats(gamma_inv_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, gamma_inv_upper68, gamma_inv_lower68 = computeStats(gamma_inv_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    gamma_inv_skew = stats.skew(gamma_inv_samples)    
    print('Mean gamma_inv=',gamma_inv_bar, ' Med gamma_inv=', gamma_inv_med, 'Skew gamma_inv=', gamma_inv_skew)

    R0_bar, R0_med, R0_std, R0_upper95, R0_lower95 = computeStats(R0_samples, bound_type='Quantiles', bound_param = [0.025, 0.975])
    _, _, _, R0_upper68, R0_lower68 = computeStats(R0_samples, bound_type='Quantiles', bound_param = [0.155, 0.835])
    R0_skew = stats.skew(R0_samples)    
    print('Mean tc=',R0_bar, ' Med tc=', R0_med, 'Skew tc=', R0_skew)

    ##### Stats on outputs of model #####
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

    # Store stats
    worksheet        = kwargs['worksheet']
    row_num          = kwargs['row_num']
    beta_stats       = [beta_bar, beta_upper95, beta_lower95, beta_upper68, beta_lower68]
    gamma_inv_stats  = [gamma_inv_bar, gamma_inv_upper95, gamma_inv_lower95, gamma_inv_upper68, gamma_inv_lower68]
    R0_stats         = [R0_bar, R0_upper95, R0_lower95, R0_upper68, R0_lower68]
    tc_stats         = [tc_bar, tc_kde_upper95, tc_kde_lower95, tc_kde_upper68, tc_kde_lower68]
    Ipeak_stats      = [Ipeak_bar, Ipeak_kde_upper95, Ipeak_kde_lower95, Ipeak_kde_upper68, Ipeak_kde_lower68]
    Tend_stats       = [Tend_bar, Tend_kde_upper95, Tend_kde_lower95, Tend_kde_upper68, Tend_kde_lower68]
    worksheet.write_row(row_num, 0,  beta_stats)
    worksheet.write_row(row_num, 5,  gamma_inv_stats)
    worksheet.write_row(row_num, 10, R0_stats)    
    worksheet.write_row(row_num, 15, tc_stats)
    worksheet.write_row(row_num, 20, Ipeak_stats)
    worksheet.write_row(row_num, 25, Tend_stats)
    

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
    X = SIR_params; 
    # X = np.vstack((beta_samples, gamma_inv_samples, R0_samples)).T
    # X = np.vstack((beta_samples, gamma_inv_samples, R0_samples, beta_samples**2, gamma_inv_samples**2)).T
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
    print('Ipeak Coeff:', regr_Ipeak.coef_, 't_c Intercept:', regr_Ipeak.intercept_)

    regr_Tend = linear_model.LinearRegression()
    Y = Tend_samples    
    regr_Tend.fit(X, Y)
    print('Tend Coeff:', regr_Tend.coef_, 'Tend Intercept:', regr_Tend.intercept_)

    # Compute similarities (Could make this a matrix)
    d_R0tc      = hyperplane_similarity(regr.coef_,regr.intercept_,regr_tc.coef_[0],regr_tc.intercept_[0])
    d_R0IPeak   = hyperplane_similarity(regr.coef_,regr.intercept_,regr_Ipeak.coef_[0],regr_Ipeak.intercept_[0])
    d_R0Tend    = hyperplane_similarity(regr.coef_,regr.intercept_,regr_Tend.coef_[0],regr_Tend.intercept_[0])
    d_tcIpeak   = hyperplane_similarity(regr_tc.coef_[0],regr_tc.intercept_[0],regr_Ipeak.coef_[0],regr_Ipeak.intercept_[0])
    d_tcTend    = hyperplane_similarity(regr_tc.coef_[0],regr_tc.intercept_[0],regr_Tend.coef_[0],regr_Tend.intercept_[0])
    d_IpeakTend = hyperplane_similarity(regr_Ipeak.coef_[0],regr_Ipeak.intercept_[0],regr_Tend.coef_[0],regr_Tend.intercept_[0])

    print('d_R0tc = ',d_R0tc, ' d_R0IPeak=', d_R0IPeak, ' d_R0Tend=', d_R0Tend)
    print('d_tcIpeak = ',d_tcIpeak, ' d_tcTend=', d_tcTend, ' d_IpeakTend=', d_IpeakTend)

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
        ax0.plot(x_vals_tc,tc_kde_pdf,'k', lw=1, label=r"kde")
        ax0.plot([tc_kde_median]*10,np.linspace(0,count[np.argmax(count)], 10),'k--', lw=2,label=r"med")
        ax0.plot([tc_kde_lower95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=1.5, label=r"Q[95\%]")
        ax0.plot([tc_kde_upper95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=1.5)
        ax0.plot([tc_kde_lower68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=1.5, label=r"Q[68\%]")
        ax0.plot([tc_kde_upper68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=1.5)

        # Plot raw mean and quantile stats
        ax0.plot([tc_bar[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r', lw=2,label=r"mean")
        if plot_data_quant:
            ax0.plot([tc_lower95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r:', lw=1.5, label=r"Q[95\%]")
            ax0.plot([tc_upper95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r:', lw=1.5)
            ax0.plot([tc_lower68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r-.', lw=1.5, label=r"Q[68\%]")
            ax0.plot([tc_upper68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'r-.', lw=1.5)
        ax0.set_title(r"Critical point $t_c$", fontsize=20)
        ax0.grid(True, alpha=0.3)
        ax0.set_xlabel(r"$t_c$", fontsize=20)
        legend = ax0.legend(fontsize=8)
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
        ax1.plot(x_vals_Ipeak,Ipeak_kde_pdf,'k', lw=1, label=r"kde")
        ax1.plot([Ipeak_kde_median]*10,np.linspace(0,count[np.argmax(count)], 10),'k--', lw=2,label=r"med")
        ax1.plot([Ipeak_kde_lower95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=1.5, label=r"Q[95\%]")
        ax1.plot([Ipeak_kde_upper95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=1.5)
        ax1.plot([Ipeak_kde_lower68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=1.5, label=r"Q[68\%]")
        ax1.plot([Ipeak_kde_upper68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=1.5)

        # Plot raw median and quantile stats
        ax1.plot([Ipeak_bar[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g', lw=2,label=r"mean")    
        if plot_data_quant:
            ax1.plot([Ipeak_lower95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g:', lw=1.5, label=r"Q[95\%]")
            ax1.plot([Ipeak_upper95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g:', lw=1.5)
            ax1.plot([Ipeak_lower68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g-.', lw=1.5, label=r"Q[68\%]")
            ax1.plot([Ipeak_upper68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'g-.', lw=1.5)

        ax1.set_title(r"Peak Infectedes $I_{peak}$", fontsize=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel(r"$I_{peak}$", fontsize=20)
        legend = ax1.legend(fontsize=8)
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
        ax2.plot(x_vals_Tend,Tend_kde_pdf,'k', lw=1, label=r"kde")    
        ax2.plot([Tend_kde_median]*10,np.linspace(0,count[np.argmax(count)], 10),'k--', lw=2,label=r"med")
        ax2.plot([Tend_kde_lower95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=1.5, label=r"Q[95\%]")
        ax2.plot([Tend_kde_upper95]*10,np.linspace(0,count[np.argmax(count)], 10),'k:', lw=1.5)
        ax2.plot([Tend_kde_lower68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=1.5, label=r"Q[68\%]")
        ax2.plot([Tend_kde_upper68]*10,np.linspace(0,count[np.argmax(count)], 10),'k-.', lw=1.5)

        # Plot raw median and quantile stats 
        ax2.plot([Tend_bar[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b', lw=2,label=r"mean")    
        if plot_data_quant:
            ax2.plot([Tend_lower95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b:', lw=1.5, label=r"Q[95\%]")
            ax2.plot([Tend_upper95[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b:', lw=1.5)
            ax2.plot([Tend_lower68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b-.', lw=1.5, label=r"Q[68\%]")
            ax2.plot([Tend_upper68[0]]*10,np.linspace(0,count[np.argmax(count)], 10),'b-.', lw=1.5)

        ax2.set_title(r"Total cases @ $t_{end}$", fontsize=20)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel(r"$T(t=$end$)$", fontsize=20)
        legend = ax2.legend(fontsize=10)
        legend.get_frame().set_alpha(0.5)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
        for tick in ax2.yaxis.get_major_ticks():
                tick.label.set_fontsize(15) 

        fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
        fig.set_size_inches(27/2, 9/2, forward=True)
        if kwargs['store_plots']:
            plt.savefig(kwargs['file_extension'] + "_CriticalPointsHistograms.png", bbox_inches='tight')


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


def plotInfected_evolution(Ivariables, Plotoptions, **kwargs):
    # Unpacking variables
    I, t = Ivariables
    T_limit, plot_peaks, x_axis_offset, y_axis_offset = Plotoptions 


    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    filename       = kwargs['file_extension']        
    store_plots    = kwargs['store_plots']
    
    if 'number_scaling' in kwargs:
        number_scaling = kwargs['number_scaling']
    else:
        number_scaling = 'million'

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
        scale     = 1 
        number_scaling = ""
    elif number_scaling == 'fraction':
        scale     = N 
        number_scaling = "fraction"

    if plot_peaks:
        tc, _, _, _, _  =  getCriticalPointsAfterPeak(I_plot)
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


############################################################################################################
#############                                PLOTS FOR SIR MODEL                               #############
############################################################################################################
def plotSIR_evolution(SIRvariables, Plotoptions, **kwargs):
    
    # Unpacking variables
    S, I, R, T, t = SIRvariables
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset = Plotoptions 
    
    figure_title   = kwargs['figure_title']
    N              = kwargs['N']
    filename       = kwargs['file_extension']
    if 'number_scaling' in kwargs:
        number_scaling = kwargs['number_scaling']
    else:
        number_scaling = 'million'

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
    elif number_scaling == 'fraction':
        scale     = N 
        number_scaling = "fraction"

    
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
        txt1 = "{per:2.3f} {number_scaling} total cases as $t(end)$"
        ax1.text(t[-1] - x_axis_offset, (total_cases/N) + y_axis_offset, txt1.format(per=total_cases/scale,number_scaling=number_scaling), fontsize=20, color='r')    

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

    if 'number_scaling' in kwargs:
        number_scaling = kwargs['number_scaling']
    else:
        number_scaling = 'million'


    # Defining scaling for text     
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
    elif number_scaling == 'fraction':
        scale     = N 
        number_scaling = "fraction"


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
        txt1 = "{per:2.4f} {number_scaling} total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (total_cases/N), txt1.format(per=total_cases/scale, number_scaling= number_scaling), fontsize=18, color='m')

        total_cases     = T_minus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} {number_scaling} total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, 0.98*(total_cases/N), txt1.format(per=total_cases/scale, number_scaling= number_scaling), fontsize=18, color='m')

        total_cases     = T_plus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.4f} {number_scaling} total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(per=total_cases/scale, number_scaling = number_scaling), fontsize=18, color='m')
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
                txt_title = r"Local peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(tc+ 5, I_tc/N , txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 30, 0.9*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, number_scaling=number_scaling, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+ 5,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, number_scaling=number_scaling, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            else:
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(tc- 40, I_tc/N , txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 20, I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, number_scaling=number_scaling, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx -30,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, number_scaling=number_scaling, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        else:
            # Adjust automatically
            txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
            ax1.text(tc+2, (1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_plus:        
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx-25, (1 + 10*scale_offset)*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, number_scaling=number_scaling, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_minus:
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+2, (1 - 10*scale_offset)*I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, number_scaling=number_scaling, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(tc, T_tc/N,'mo', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:5.5f} {number_scaling} by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="m", bbox=dict(facecolor='white', alpha=0.75))


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


def plotSIR_evolutionStochastic(S_variables, I_variables, R_variables, T_variables, Plotoptions, **kwargs):    

    S       = S_variables[0,:]
    S_med   = S_variables[1,:]
    I       = I_variables[0,:]
    I_med   = I_variables[1,:]
    R       = R_variables[0,:]
    R_med   = R_variables[1,:]
    T       = T_variables[0,:]
    T_med   = T_variables[1,:]



    if 'do_std' in kwargs and kwargs["do_std"]:
        S_plus  = S + 2*S_variables[4,:]
        S_minus = S - 2*S_variables[4,:]
        I_plus  = I + 2*I_variables[4,:]
        I_minus = I - 2*I_variables[4,:]
        R_plus  = R + 2*R_variables[4,:]
        R_minus = R - 2*R_variables[4,:]
        T_plus  = T + 2*T_variables[4,:]
        T_minus = T - 2*T_variables[4,:]        
    else:        
        S_plus  = S_variables[2,:]
        S_minus = S_variables[3,:]
        I_plus  = I_variables[2,:]
        I_minus = I_variables[3,:]
        R_plus  = R_variables[2,:]
        R_minus = R_variables[3,:]    
        T_plus  = T_variables[2,:]
        T_minus = T_variables[3,:]


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
    store_plots    = kwargs['store_plots']

    if 'x_tick_names' in kwargs:
        x_tick_names   = kwargs['x_tick_names']
        x_tick_numbers = np.arange(0, len(S), kwargs['x_tick_step'])
    
    if 'text_error' in kwargs:
        text_error = kwargs['text_error']
    else:
        text_error = ''

    if 'number_scaling' in kwargs:
        number_scaling = kwargs['number_scaling']
    else:
        number_scaling = 'million'


    # Defining scaling for text     
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
    elif number_scaling == 'fraction':
        scale     = N 
        number_scaling = "fraction"


    # Plot the data of three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()
    fig.suptitle(figure_title,fontsize=25)    

    # Variable evolution
    if show_S:
        # ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.35)
        ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
        ax1.plot(t, S_med/N, 'k--', lw=2, alpha=0.55)
        ax1.fill_between(t,(S_minus)/N,(S_plus)/N, color='k', alpha=0.15)

    # ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, I/N, 'r', lw=2,   label='Infected Cases')
    ax1.plot(t, I_med/N, 'r--', lw=2, alpha=0.55)
    # ax1.plot(t, I_minus/N, 'r:', lw=2, alpha=0.25)
    # ax1.plot(t, I_plus/N, 'r:', lw=2, alpha=0.25)
    # ax1.fill_between(t,(I - I_std)/N,(I + I_std)/N, color='r', alpha=0.10)
    if 'do_std' in kwargs and kwargs["do_std"]:
        ax1.fill_between(t,(I_minus)/N,(I_plus)/N, color='r', alpha=0.10, label=r'$\pm2\sigma$')
    else:
        ax1.fill_between(t,(I_minus)/N,(I_plus)/N, color='r', alpha=0.10, label=r'$Q[95\%]$')

    scenario = 2
    if show_T:
        # ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
        ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
        ax1.plot(t, T_med/N, 'm--', lw=2,  alpha=0.55)
        # ax1.plot(t, T_minus/N, 'm:',  lw=2, alpha=0.25)
        # ax1.plot(t, T_plus/N, 'm:',  lw=2, alpha=0.25)
        # ax1.fill_between(t,(T - T_std)/N,(T + T_std)/N, color='m', alpha=0.10)
        ax1.fill_between(t,(T_minus)/N,(T_plus)/N, color='m', alpha=0.10)

        total_cases     = T[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        txt1 = "{per:2.3f} {number_scaling} total cases as $t(end)$"
        ax1.text(t[-1]-x_axis_offset, 1.02*(total_cases/N), txt1.format(number_scaling =number_scaling,  per=total_cases/scale), fontsize=15, color='m')

        total_cases     = T_minus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        # txt1 = "{per:2.4f} {number_scaling} total cases as $t(end)$"
        # ax1.text(t[-1]-x_axis_offset, 0.98*(total_cases/N), txt1.format(number_scaling =number_scaling,per=total_cases/scale), fontsize=12, color='m')

        total_cases     = T_plus[-1]
        print('Total Cases when growth linear = ', total_cases)
        # ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'k--')
        # txt1 = "{per:2.4f} {number_scaling} total cases as $t(end)$"
        # ax1.text(t[-1]-x_axis_offset, (1 + scale_offset)*(total_cases/N), txt1.format(number_scaling =number_scaling, per=total_cases/scale), fontsize=12, color='m')
        # fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
        if show_S:
            ax1.text(0, 0.5, text_error, fontsize=18, bbox=dict(facecolor='red', alpha=0.1))
        else:
            ax1.text(0, 0.8, text_error, fontsize=18, bbox=dict(facecolor='red', alpha=0.1))
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


    do_plus = 0
    do_minus = do_plus
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
                txt_title = r"Local peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(tc+ 5, I_tc/N , txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 30, 0.9*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, number_scaling=number_scaling, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Local peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+ 5,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, number_scaling=number_scaling, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            else:
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(tc- 40, I_tc/N , txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx- 20, I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, number_scaling=number_scaling, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
                txt_title = r"Peak infected: {I_tc:5.5f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx -30,I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, number_scaling=number_scaling, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        else:
            # Adjust automatically
            txt_title = r"Peak infected: {I_tc:2.3f} {number_scaling} by day {peak_days:10.0f} " 
            ax1.text(tc+10, (1)*I_tc/N , txt_title.format(I_tc=I_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_plus:        
                txt_title = r"Peak infected: {I_tc:2.3f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_plus_idx-25, 1.05*I_tc_plus/N, txt_title.format(I_tc=I_tc_plus/scale, number_scaling=number_scaling, peak_days= I_tc_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
            if do_minus:
                txt_title = r"Peak infected: {I_tc:2.3f} {number_scaling} by day {peak_days:10.0f} " 
                ax1.text(I_tc_minus_idx+10, I_tc_minus/N, txt_title.format(I_tc=I_tc_minus/scale, number_scaling=number_scaling, peak_days= I_tc_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(tc, T_tc/N,'mo', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:2.3f} {number_scaling} by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, number_scaling=number_scaling, peak_days= tc), fontsize=20, color="m", bbox=dict(facecolor='white', alpha=0.75))


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
        if 'do_std' in kwargs and kwargs["do_std"]:
            plt.savefig(filename + "_std.png", bbox_inches='tight')
            # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')
        else:
            plt.savefig(filename + ".png", bbox_inches='tight')
            # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')


def plotSIR_sampledParams(beta_samples, gamma_inv_samples, filename, *prob_params):
    fig, (ax1,ax2) = plt.subplots(1,2, constrained_layout=True)

    ###########################################################
    ################## Plot for Beta Samples ##################
    ###########################################################
    count, bins, ignored = ax1.hist(beta_samples, 30, density=True, alpha=0.55, edgecolor='k')

    if prob_params[0] == 'uniform':
        ax1.set_xlabel(r"$\beta \sim \mathcal{U}$", fontsize=15)        
        ax1.set_xlim(0, 1.0)    

    if prob_params[0] == 'gaussian':
        mu    = prob_params[1]
        sigma = prob_params[2] + 0.00001
        bins  = np.arange(0,1,0.001)
        ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        ax1.set_xlabel(r"$\beta \sim \mathcal{N}$", fontsize=15)    
        ax1.set_xlim(0, 1.0)    

    if prob_params[0] == 'gamma':
        g_dist    = gamma_dist(prob_params[2], prob_params[1], prob_params[3])
        # Plot gamma samples and pdf
        x = np.arange(0,1,0.001)
        ax1.plot(x, g_dist.pdf(x), 'r',label=r'$k = 1, \mu=%.1f,\ \theta=%.1f$' % (prob_params[1], prob_params[2]))
        ax1.set_xlabel(r"$\beta \sim Gamma$", fontsize=15)    

    if prob_params[0] == 'log-normal':
        mu    = prob_params[1]
        sigma = prob_params[2] + 0.00001 
        # x = np.linspace(min(bins), max(bins), 10000)
        x = np.arange(0,1,0.001)
        pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))        
        ax1.plot(x, pdf, linewidth=2, color='r')
        ax1.set_xlabel(r"$\beta \sim LogNormal$", fontsize=15)    

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(15) 
    plt.xlim(0, 1.0)            
    ax1.grid(True, alpha=0.3)
    ax1.set_title(r"Distribution of $\beta$ samples", fontsize=20)
    
    ###############################################################
    ################## Plot for Gamma^-1 Samples ##################
    ###############################################################
    count, bins, ignored = ax2.hist(gamma_inv_samples, 30, density=True, alpha=0.55, edgecolor='k')
    if prob_params[0] == 'gaussian':
        mu    = prob_params[3]
        sigma = prob_params[4] + 0.00001
        ax2.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')

        ax2.set_xlabel(r"$\gamma^{-1} \sim \mathcal{N}$", fontsize=15)
    
    if prob_params[0] == 'uniform':
        ax2.set_xlabel(r"$\gamma^{-1} \sim \mathcal{U}$", fontsize=15)          

    if prob_params[0] == 'gamma':
        g_dist    = gamma_dist(prob_params[5], prob_params[4], prob_params[6])
        # Plot gamma samples and pdf
        x = np.arange(1,15,0.1)
        ax2.plot(x, g_dist.pdf(x), 'r',label=r'$k = 1, \mu=%.1f,\ \theta=%.1f$' % (prob_params[3], prob_params[4]))
        ax2.set_xlabel(r"$\gamma^{-1} \sim Gamma$", fontsize=15)          

    if prob_params[0] == 'log-normal':
        mu    = prob_params[3]
        sigma = prob_params[4] + 0.00001
        x = np.arange(1,15,0.1)
        pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
        plt.plot(x, pdf, linewidth=2, color='r')
        ax2.set_xlabel(r"$\gamma^{-1} \sim LogNormal$", fontsize=15)          


    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(15) 
    for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)  
    plt.xlim(1, 17) 
    ax2.grid(True, alpha=0.3)    
    plt.title(r"Distribution of $\gamma^{-1}$ samples", fontsize=20)    

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(20/2, 8/2, forward=True)    
    
    # Store plot
    plt.savefig(filename + ".png", bbox_inches='tight')


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

