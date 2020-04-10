import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from   scipy.optimize import fsolve
import pandas as pd
from scipy.signal import find_peaks


def plotInfected_evolution(txt_title, SIRparams, Ivariables, Plotoptions, filename, number_scaling = 'million', x_tick_labels = [], infected_data = np.array([]), infected_estimated_data = np.array([]), check_points = np.array([]), pi_t = np.array([])):
    scenario, r0, beta, gamma_inv, N = SIRparams
    I, t = Ivariables
    T_limit, plot_peaks, x_axis_offset, y_axis_offset, store_plots = Plotoptions 

    I_plot = I[0:T_limit]
    t_plot = t[0:T_limit]

    # Check if x_tick_labels is given
    x_tick_numbers, x_tick_names = x_tick_labels
    
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
    fig.suptitle(txt_title.format(scenario=scenario, R0=float(r0), beta= beta, gamma = 1/gamma_inv),fontsize=20)

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
        txt_title = r"Peak infected: {I_tc:2.4f} {number_scaling} by day {peak_days:10.0f} " 
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

    # if pi_t.size > 0:
    #     ax1a = ax1.twinx()
        # color = 'tab:blue'
        # ax1a.set_ylabel('$\pi(t)$', color=color)  # we already handled the x-label with ax1
        # ax1a.plot(t_plot, pi_t[0:T_limit], color=color)
        # ax1a.tick_params(axis='y', labelcolor=color)

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
            ax2.text(max_peaks[0]+x_axis_offset, 1.5*I_plot[max_peaks[0]]/N, txt_title.format(I_tc=I_plot[max_peaks[0]]/scale, number_scaling=number_scaling,  peak_days= max_peaks[0]), fontsize=15, color="r",  bbox=dict(facecolor='white', alpha=0.75))
        if min_peaks.size > 0:
            ax2.plot(min_peaks[0], I_plot[min_peaks[0]]/N, 'ro',  markersize=5,  alpha= 0.5)
            ax2.text(min_peaks[0]+x_axis_offset, 0.5*I_plot[min_peaks[0]]/N, txt_title.format(I_tc=I_plot[min_peaks[0]]/scale, number_scaling=number_scaling,  peak_days= min_peaks[0]), fontsize=15, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if check_points.size > 0:
        txt_title = r"Infected: {I_tc:2.4f} {number_scaling} by day {peak_days:10.0f} " 
        ax2.plot(check_points, I_plot[check_points]/N,'k+', markersize=8, alpha = 0.5, label='Check-points')    
        for ii in range(check_points.size):
            ax2.text(check_points[ii]+2, I_plot[check_points[ii]]/N, txt_title.format(I_tc=I_plot[check_points[ii]]/scale, number_scaling=number_scaling,  peak_days= check_points[ii]), fontsize=11.5, color="k", bbox=dict(facecolor='white', alpha=0.75))
    
    plt.yscale("log")

    ax2.set_xlabel('Time /days', fontsize=20)
    ax2.set_ylabel('Fraction of Population', fontsize=20)
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
        plt.savefig(filename + "_infected.png", bbox_inches='tight')
        # plt.savefig(filename + "_infected.pdf", bbox_inches='tight')


def getCriticalPointsAfterPeak(I):
    tc    =  np.argmax(I)
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


def plotSIR_evolution(txt_title, SIRparams, SIRvariables, Plotoptions, filename, number_scaling = 'million', x_tick_labels = []):
    scenario, r0, beta, gamma_inv, N = SIRparams
    S, I, R, T, t = SIRvariables
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, store_plots = Plotoptions 
    
    # Check if x_tick_labels is given
    x_tick_numbers, x_tick_names = x_tick_labels
    
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
        I_tc        = I[tc]
        T_tc  = T[tc]
    
    total_cases     = T[-1]

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()    
    fig.suptitle(txt_title.format(scenario=scenario, R0=float(r0), beta= beta, gamma = 1/gamma_inv),fontsize=20)

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
        ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
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
    plt.xticks(x_tick_numbers, x_tick_names)
    legend = ax1.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(20)     

    plt.grid(True, color='k', alpha=0.2, linewidth = 0.25)        
    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)
    
    if store_plots:
        plt.savefig(filename + ".png", bbox_inches='tight')
        # plt.savefig(filename + ".pdf", bbox_inches='tight')


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



def plotSIR_evolutionErrors(txt_title, SIRparams, S_variables, I_variables, R_variables, Plotoptions, text_error, store_plots, filename):
    scale = 1000000        

    # Unpack
    scenario, r0, beta, gamma_inv, N = SIRparams
    plot_all, show_S, show_T, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, beta_error, scale_offset = Plotoptions 

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
    ax1.plot(t, I/N, 'r', lw=2,   label='Number of Infected People')
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
        ax1.text(0.8*Tf, 0.5, text_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))
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
            ax1.plot(tc, T_tc/N,'ro', markersize=8)
            txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
            ax1.text(tc+10, T_tc/N, txt_title2.format(peak_total=T_tc/scale, peak_days= tc), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

    # ####### OPTIONAL STUFF #######  
    # if show_analytic_limit:
    #     ax1.plot(t, covid_SinfS0*np.ones(len(t)), 'm--')
    #     txt1 = "{per:2.2f} population infected"
    #     ax1.text(t[0], covid_SinfS0 - 0.05, txt1.format(per=covid_SinfS0[0]), fontsize=20, color='m')


    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
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
        plt.savefig(filename + "_all.png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')


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

