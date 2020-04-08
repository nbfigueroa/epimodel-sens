import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from   scipy.optimize import fsolve

def plotSIR_evolution(txt_title, SIRparams, SIRvariables, Plotoptions, store_plots, filename):
    scenario, r0, beta, gamma_inv, N = SIRparams
    S, I, R, T, t = SIRvariables
    plot_all, show_S, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset = Plotoptions 
    
    total_cases     = T[-1]

    if plot_peaks:
        peak_inf_idx =  np.argmax(I)
        peak_inf     = I[peak_inf_idx]
        peak_total_inf  = T[peak_inf_idx]

     # These could be options for adaptive scaling
    scale = 1000000

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax1 = plt.subplots()    
    fig.suptitle(txt_title.format(scenario=scenario, R0=float(r0), beta= beta, gamma = 1/gamma_inv),fontsize=20)

    # Variable evolution    
    ax1.plot(t, I/N, 'r',   lw=2,   label='Infected')

    if plot_all:        
        ax1.plot(t, T/N, 'y', lw=2,   label='Total Cases')
        # Plot Final Epidemic Size
        if show_analytic_limit:
            ax1.plot(t, One_SinfN*np.ones(len(t)), 'm--')
            txt1 = "Analytic Epidemic Size: 1-S(inf)/N={per:2.2f} percentage (analytic)"
            ax1.text(t[-1]-200, One_SinfN + 0.02, txt1.format(per=One_SinfN[0]), fontsize=20, color='m')

        ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
        txt1 = "{per:2.3f} million total cases as $t(end)$"
        ax1.text(t[-1] - x_axis_offset, (total_cases/N) - y_axis_offset, txt1.format(per=total_cases/scale), fontsize=20, color='r')    
    
    if show_S:
        ax1.plot(t, S/N, 'k',   lw=2, label='Susceptible')
    
    if show_R:
        ax1.plot(t, R/N, 'g--',  lw=1,  label='Recovered')

    if plot_peaks:
        # Plot peak points
        ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)
        
        txt_title = r"Peak infected: {peak_inf:2.4f}million by day {peak_days:10.0f} " 
        txt_title2 = r"Total Cases: {peak_total:2.4f}million by day {peak_days:10.0f} " 
        ax1.text(peak_inf_idx+10, peak_inf/N, txt_title.format(peak_inf=peak_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

        if plot_all == 1:
            ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
            ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

    # Making things beautiful
    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax1.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

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

        peak_infe_idx =  np.argmax(Inf)
        peak_infe     = Inf[peak_infe_idx]

        peak_inf_idx =  np.argmax(I)
        peak_inf     = I[peak_inf_idx]

        if show_Q:
            peak_Q_idx =  np.argmax(Q)
            peak_Q     = Q[peak_Q_idx]

        if show_E:
            peak_E_idx =  np.argmax(E)
            peak_E     = E[peak_E_idx]

        if plot_all:
            peak_total_inf  = T[peak_inf_idx]


    fig, ax1 = plt.subplots()
    fig.suptitle(txt_title.format(R0=r0, beta= beta, gamma_inv = gamma_inv, sigma_inv = sigma_inv, tau_q_inv = tau_q_inv, q=q),fontsize=15)

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
    ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)
    txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f}" 
    txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f}" 
    ax1.text(peak_inf_idx+10, peak_inf/N, txt_title.format(peak_inf=peak_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if plot_all:
        ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
        ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))
        
    if plot_peaks:        
        # Plot peak points
        ax1.plot(peak_infe_idx, peak_infe/N,'ro', markersize=8)
        txt_title3 = r"Peak Infectuous (I+Q): {peak_infe:5.5f}million by day {peake_days:10.0f}" 
        ax1.text(peak_infe_idx+10, peak_infe/N, txt_title3.format(peak_infe=peak_infe/scale, peake_days= peak_infe_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.95))
        
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
    ax1.set_xlabel('Time /days', fontsize=20)
    ax1.set_ylabel('Fraction of Population', fontsize=20)
    ax1.yaxis.set_tick_params(length=0)
    ax1.xaxis.set_tick_params(length=0)
    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax1.legend(fontsize=20)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax1.spines[spine].set_visible(True)

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + '.png', bbox_inches='tight')
        plt.savefig(filename + '.pdf', bbox_inches='tight')



def plotSIR_evolutionErrors(txt_title, SIRparams, S_variables, I_variables, R_variables, Plotoptions, store_plots, filename):
    scale = 1000000    
    
    # Unpack
    scenario, r0, beta, gamma_inv, N = SIRparams
    plot_all, show_S, show_R, show_analytic_limit, plot_peaks, x_axis_offset, y_axis_offset, beta_error = Plotoptions 

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
    ax1.text(Tf/2, 0.9, r"$\beta \pm %1.2f \beta $"%beta_error, fontsize=20, bbox=dict(facecolor='red', alpha=0.1))

    # Variable evolution
    ax1.plot(t, S_plus/N, 'k--', lw=2, alpha=0.25)
    ax1.plot(t, S/N, 'k', lw=2, label='Susceptible')
    ax1.plot(t, S_minus/N, 'k--', lw=2, alpha=0.25)

    ax1.plot(t, I_plus/N, 'r--',  lw=2, alpha=0.25)
    ax1.plot(t, I/N, 'r', lw=2,   label='Infected')
    ax1.plot(t, I_minus/N, 'r--', lw=2, alpha=0.25)

    ax1.plot(t, T_plus/N, 'm--',  lw=2, alpha=0.25)
    ax1.plot(t, T/N, 'm',  lw=2, label='Total Cases')
    ax1.plot(t, T_minus/N, 'm--',  lw=2, alpha=0.25)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)
    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as $t(end)$"
    ax1.text(t[-1]-x_axis_offset, (total_cases/N) - 0.05, txt1.format(per=total_cases/scale), fontsize=20, color='r')

    total_cases     = T_minus[-1]
    print('Total Cases when growth linear = ', total_cases)
    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as $t(end)$"
    ax1.text(t[-1]-x_axis_offset, (total_cases/N) - 0.05, txt1.format(per=total_cases/scale), fontsize=20, color='r')

    total_cases     = T_plus[-1]
    print('Total Cases when growth linear = ', total_cases)
    ax1.plot(t, (total_cases/N)*np.ones(len(t)), 'r--')
    txt1 = "{per:2.2f} million total cases as $t(end)$"
    ax1.text(t[-1]-x_axis_offset, (total_cases/N) - 0.05, txt1.format(per=total_cases/scale), fontsize=20, color='r')
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
    peak_inf_idx =  np.argmax(I)
    peak_inf     = I[peak_inf_idx]
    print('Peak Instant. Infected = ', peak_inf,'by day=', peak_inf_idx)

    peak_total_inf  = T[peak_inf_idx]
    print('Total Cases when Peak = ', peak_total_inf,'by day=', peak_inf_idx)

    total_cases     = T[-1]
    print('Total Cases when growth linear = ', total_cases)

    peak_inf_plus_idx =  np.argmax(I_plus)
    peak_inf_plus     = I_plus[peak_inf_plus_idx]
    print('Peak Instant. Infected - Error= ', peak_inf_plus,'by day=', peak_inf_plus_idx)

    peak_inf_minus_idx =  np.argmax(I_minus)
    peak_inf_minus     = I_minus[peak_inf_minus_idx]
    print('Peak Instant. Infected + Error= ', peak_inf_minus,'by day=', peak_inf_minus_idx)

    # Plot peak points
    ax1.plot(peak_inf_idx, peak_inf/N,'ro', markersize=8)
    # Plot peak points
    ax1.plot(peak_inf_plus_idx, peak_inf_plus/N,'ro', markersize=8)
    # Plot peak points
    ax1.plot(peak_inf_minus_idx, peak_inf_minus/N,'ro', markersize=8)
    
    txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} " 
    ax1.text(peak_inf_idx+10, (1.1)*peak_inf/N , txt_title.format(peak_inf=peak_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r",  bbox=dict(facecolor='white', alpha=0.75))
    txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} " 
    ax1.text(peak_inf_plus_idx+5, (1.4)*peak_inf_plus/N, txt_title.format(peak_inf=peak_inf_plus/scale, peak_days= peak_inf_plus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))
    txt_title = r"Peak infected: {peak_inf:5.5f}million by day {peak_days:10.0f} " 
    ax1.text(peak_inf_minus_idx+5, (0.6)*peak_inf_minus/N, txt_title.format(peak_inf=peak_inf_minus/scale, peak_days= peak_inf_minus_idx), fontsize=12, color="r",  bbox=dict(facecolor='white', alpha=0.75))

    if plot_all == 1:
        ax1.plot(peak_inf_idx, peak_total_inf/N,'ro', markersize=8)
        txt_title2 = r"Total Cases: {peak_total:5.5f}million by day {peak_days:10.0f} " 
        ax1.text(peak_inf_idx+10, peak_total_inf/N, txt_title2.format(peak_total=peak_total_inf/scale, peak_days= peak_inf_idx), fontsize=20, color="r", bbox=dict(facecolor='white', alpha=0.75))

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

    fig.subplots_adjust(left=.12, bottom=.14, right=.93, top=0.93)
    fig.set_size_inches(27.5/2, 16.5/2, forward=True)

    if store_plots:
        plt.savefig(filename + "_all.png", bbox_inches='tight')
        # plt.savefig(file_extensions[0] + "_all.pdf", bbox_inches='tight')


