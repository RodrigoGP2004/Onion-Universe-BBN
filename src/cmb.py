import os
import glob
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import scipy.ndimage as ndimage
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize
from scipy.special import zeta
from matplotlib.ticker import ScalarFormatter

PLOT_EXPANSION_RATES = True
PLOT_Z_CMB = True
PLOT_TAU_XE = True
PLOT_DISTANCES = True
COMPUTE_2D_GRID = True
PLOT_CONTOURS = True

os.chdir(os.path.dirname(os.path.abspath(__file__)))

Q_NP = 1.293332       
TAU_N = 879.4     
M_E = 0.51099895       
T0 = 2.348e-10               
B_H = 13.6e-6
B_HE = 24.5876e-6                 
SIGMA_T = 6.6524587e-25
C_LIGHT = 2.99792458e10 
LAMBDA_ALPHA = 1.215682e-5 
LAMBDA_2S1S = 8.224           
KM_TO_M = 1e3
MPC_TO_M = 3.08567758e22
HBAR_C_MEV_CM = 1.9732698e-11 
M_P_MEV = 938.272

G_STAR_I = 10.75
T_EE = 0.187

H0_LCDM = 71.3 
H0_ONION = 71.3 
H0_S_LCDM = H0_LCDM * KM_TO_M / MPC_TO_M
H0_S_ONION = H0_ONION * KM_TO_M / MPC_TO_M

ETA_LCDM = 6.1395e-10
YP_LCDM = 0.2462715
YD_LCDM = 3.808996e-05
Y3HE_LCDM = 2.424953e-05   
XP_LCDM = 1.0 - YP_LCDM - YD_LCDM - Y3HE_LCDM
RS_PLANCK = 147

def calc_g_star_limits(T_nu):
    if T_nu is None or np.isnan(T_nu): 
        return 3.384, 3.938
    x = 3.0 * np.log(T_nu / 0.18)
    w = 0.5 * (1.0 + np.tanh(x))
    gs0 = w * 3.384 + (1.0 - w) * 7.25
    gss0 = w * 3.938 + (1.0 - w) * 7.25
    return gs0, gss0

def get_g_star(T_MeV, gs0=3.384):
    x = 3.0 * np.log(T_MeV / 0.18)
    return gs0 + (G_STAR_I - gs0) * 0.5 * (1.0 + np.tanh(x))

def get_g_star_s(T_MeV, gss0=3.938):
    x = 3.0 * np.log(T_MeV / 0.18)
    return gss0 + (G_STAR_I - gss0) * 0.5 * (1.0 + np.tanh(x))

def dln_g_dln_T(T_MeV, gss0=3.938):
    x = 3.0 * np.log(T_MeV / 0.18)
    th = np.tanh(x)
    dg_dlnT = (G_STAR_I - gss0) * 0.5 * 3.0 * (1.0 - th**2)
    return dg_dlnT / get_g_star_s(T_MeV, gss0)

def h_lcdm(lnT, Omega_L0=0.685):
    T = np.exp(lnT)
    h = H0_LCDM / 100.0 
    omega_r = 2.473e-5 + 1.710e-5
    omega_m = 0.143
    
    Omega_r0 = omega_r / (h**2) 
    Omega_m0 = omega_m / (h**2)
    Omega_k0 = 0.0 
    Omega_L0_calc = 1.0 - (Omega_r0 + Omega_m0)
    
    ratio = T / T0
    rad_term = Omega_r0 * (get_g_star(T, 3.384) / 3.384) * (ratio**4)
    mat_term = Omega_m0 * (get_g_star_s(T, 3.938) / 3.938) * (ratio**3)
    return H0_S_LCDM * np.sqrt(rad_term + mat_term + Omega_k0 * (ratio**2) + Omega_L0_calc)

def h_onion(lnT, Omega_L0=0.52):
    T = np.exp(lnT)
    Omega_k0 = Omega_L0 - 1.0
    return H0_S_ONION * np.sqrt(-Omega_k0 * (T / T0)**2 + Omega_L0)

def integrate_time(T_arr, h_func1, h_func2):
    lnT_inf = np.log(5e21) 
    time_1, time_2 = [], []
    for T_val in T_arr:
        lnT = np.log(T_val)
        t1, _ = quad(lambda x: (1.0 + dln_g_dln_T(np.exp(x))/3.0) / h_func1(x), lnT, lnT_inf)
        t2, _ = quad(lambda x: (1.0 + dln_g_dln_T(np.exp(x))/3.0) / h_func2(x), lnT, lnT_inf)
        time_1.append(t1)
        time_2.append(t2)
    return np.array(time_1), np.array(time_2)

def onion_time_analytic(T_arr, Omega_L0=0.97):
    T_curv = 1.0 / (H0_S_ONION * np.sqrt(Omega_L0))
    t0 = 0.5 * T_curv * np.log((1.0 + np.sqrt(Omega_L0)) / (1.0 - np.sqrt(Omega_L0)))
    return T_curv * np.arcsinh(T0 * np.sinh(t0 / T_curv) / T_arr)

def compute_event_time(T_target, h_func):
    lnT_inf = np.log(5e21)
    t, _ = quad(lambda x: (1.0 + dln_g_dln_T(np.exp(x))/3.0) / h_func(x), np.log(T_target), lnT_inf)
    return t

def saha_he_fraction(T_arr, eta_val, x_p, x_4he, x_d, x_3he):
    n_H_tot = x_p + (x_d / 2.0)
    n_He_tot = (x_4he / 4.0) + (x_3he / 3.0)
    if n_He_tot < 1e-15: return np.zeros_like(T_arr)
    
    f_H = n_H_tot / n_He_tot 
    coef = 4.0 * (np.pi**2) / (2.0 * zeta(3) * eta_val * n_He_tot)
    mass_term = (M_E / (2.0 * np.pi * T_arr))**1.5
    
    with np.errstate(over='ignore', under='ignore'):
        S_T = coef * mass_term * np.exp(-B_HE / T_arr)
    
    b = f_H + S_T
    x_he = (-b + np.sqrt(b**2 + 4.0 * S_T)) / 2.0 
    return np.clip(np.nan_to_num(x_he, nan=0.0), 0.0, 1.0)

def saha_h_fraction(T_MeV, eta_val, x_p, x_4he, x_d, x_3he):
    n_H_tot = x_p + (x_d / 2.0)
    if n_H_tot < 1e-15: return 1.0 
    coef = (np.pi**2) / (2.0 * zeta(3) * eta_val * n_H_tot)
    S_T = coef * (M_E / (2.0 * np.pi * T_MeV))**1.5 * np.exp(-B_H / T_MeV)
    return (-S_T + np.sqrt(S_T**2 + 4.0 * S_T)) / 2.0

def peebles_dxedx(x, x_e_arr, eta_val, x_p, x_4he, x_d, x_3he, h_func):
    x_e = max(1e-15, min(1.0, x_e_arr[0]))
    T_MeV = B_H / x
    h_s = h_func(np.log(T_MeV))
    
    n_gamma = (2.0 * zeta(3) / np.pi**2) * (T_MeV / HBAR_C_MEV_CM)**3
    n_H_tot = eta_val * n_gamma * (x_p + (x_d / 2.0))
    if n_H_tot < 1e-30: return [0.0]
    
    T_K = T_MeV * 1.16045e10
    T4 = T_K / 10000.0
    alpha_b = 1.14 * 4.309e-13 * (T4**-0.6166) / (1.0 + 0.6703 * (T4**0.53))
    
    mass_cm3 = (M_E * T_MeV / (2.0 * np.pi))**1.5 / (HBAR_C_MEV_CM**3)
    beta_b = alpha_b * mass_cm3 * np.exp(-x / 4.0)
    
    n_H_neutral = n_H_tot * max(1.0 - x_e, 1e-15) 
    r_lya = 1e30 if n_H_neutral < 1e-30 else (8.0 * np.pi * h_s) / (3.0 * (LAMBDA_ALPHA**3) * n_H_neutral)
    
    rate_down_eff = 3.0 * r_lya + LAMBDA_2S1S
    c_factor = rate_down_eff / (beta_b + rate_down_eff)
    dx_e_dt = -c_factor * (n_H_tot * (x_e**2) * alpha_b - (1.0 - x_e) * beta_b * np.exp(-0.75 * x))
    time_factor = 1.0 + dln_g_dln_T(T_MeV) / 3.0
    return [(dx_e_dt * time_factor) / (x * h_s)]

def solve_peebles_recombination(eta_val, x_p, x_4he, x_d, x_3he, h_func):
    T_start, T_end = 0.40e-6, 0.0005e-6  
    x_start, x_end = B_H / T_start, B_H / T_end
    x_e_init = saha_h_fraction(T_start, eta_val, x_p, x_4he, x_d, x_3he)
    
    sol = solve_ivp(
        peebles_dxedx, [x_start, x_end], [x_e_init], method='BDF',
        t_eval=np.linspace(x_start, x_end, 100000), 
        args=(eta_val, x_p, x_4he, x_d, x_3he, h_func), 
        rtol=1e-6, atol=1e-8
    )
    if not sol.success: 
        return lambda T: np.ones_like(T) if isinstance(T, np.ndarray) else 1.0
    
    T_arr = B_H / np.asarray(sol.t)
    return interp1d(T_arr[::-1], sol.y[0][::-1], kind='linear', bounds_error=False, fill_value=(sol.y[0][-1], 1.0))

def compute_cmb_dynamics(T_arr, lnT_arr, interp_xe, eta_val, x_p, x_4he, x_d, x_3he, h_func):
    h_s = np.array([h_func(lnt) for lnt in lnT_arr])
    x_he = saha_he_fraction(T_arr, eta_val, x_p, x_4he, x_d, x_3he) 
    
    n_b = eta_val * (2.0 * zeta(3) / np.pi**2) * (T_arr / HBAR_C_MEV_CM)**3
    n_H_tot = n_b * (x_p + (x_d / 2.0))
    n_He_tot = n_b * ((x_4he / 4.0) + (x_3he / 3.0)) 
    
    n_e = np.maximum(n_H_tot * np.maximum(0.0, interp_xe(T_arr)) + n_He_tot * x_he, 1e-30)
    
    time_factor = 1.0 + dln_g_dln_T(T_arr) / 3.0
    dtau_dlnT = (C_LIGHT * SIGMA_T * n_e * time_factor) / h_s
    tau_arr = cumulative_trapezoid(dtau_dlnT, lnT_arr, initial=0)
    
    idx_tau1 = np.argmin(np.abs(tau_arr - 1.0))
    idx_peak = np.argmax(dtau_dlnT * np.exp(-tau_arr))
    return T_arr[idx_tau1], T_arr[idx_peak], tau_arr

def compute_sound_horizon(T_target, eta_val, h_func, gss0):
    def integrand(lnT):
        T = np.exp(lnT)
        gs_ratio = get_g_star_s(T, gss0) / gss0
        a_T = (T0 / T) * (1.0 / gs_ratio)**(1.0/3.0) 
        
        n_gamma = (2.0 * zeta(3) / np.pi**2) * (T / HBAR_C_MEV_CM)**3 
        rho_gamma = (np.pi**2 / 15.0) * (T**4) / (HBAR_C_MEV_CM**3) 
        rho_b = eta_val * n_gamma * M_P_MEV * gs_ratio 
        
        cs = C_LIGHT / np.sqrt(3.0 * (1.0 + (3.0 * rho_b) / (4.0 * rho_gamma))) 
        time_factor = 1.0 + dln_g_dln_T(T, gss0) / 3.0
        return (cs / a_T) * time_factor / h_func(lnT) 
    
    rs_cm, _ = quad(integrand, np.log(T_target), np.log(5e21)) 
    return rs_cm / (MPC_TO_M * 100.0)

def compute_comoving_distance(T_target, h_func, gss0):
    def integrand(lnT):
        T = np.exp(lnT)
        a_T = (T0 / T) * (1.0 / (get_g_star_s(T, gss0) / gss0))**(1.0/3.0)
        time_factor = 1.0 + dln_g_dln_T(T, gss0) / 3.0
        return C_LIGHT * time_factor / (a_T * h_func(lnT))
    
    chi_cm, _ = quad(integrand, np.log(T0), np.log(T_target))
    return chi_cm / (MPC_TO_M * 100.0)

def compute_dm_onion(chi_mpc, Omega_L0):
    if Omega_L0 < 1e-8:
        R0_cm = C_LIGHT / H0_S_ONION
    else:
        T_curv = 1.0 / (H0_S_ONION * np.sqrt(Omega_L0))
        t0_s = T_curv * np.arctanh(np.sqrt(Omega_L0))
        R0_cm = C_LIGHT * T_curv * np.sinh(t0_s / T_curv)
        
    R0_mpc = R0_cm / (MPC_TO_M * 100.0)
    return np.abs(R0_mpc * np.sin(chi_mpc / R0_mpc))

def compute_dm_onion_analytic(z_arr, Omega_L0):
    if Omega_L0 < 1e-8:
        R0_mpc = (C_LIGHT / H0_S_ONION) / (MPC_TO_M * 100.0)
        dc_mpc = R0_mpc * np.log(1.0 + z_arr)
        return np.abs(R0_mpc * np.sin(dc_mpc / R0_mpc))
    else:
        T_curv = 1.0 / (H0_S_ONION * np.sqrt(Omega_L0))
        t0_Tcurv = np.arctanh(np.sqrt(Omega_L0)) 
        R0_mpc = (C_LIGHT * T_curv * np.sinh(t0_Tcurv)) / (MPC_TO_M * 100.0)
        
        tz_Tcurv = np.arcsinh(np.sinh(t0_Tcurv) / (1.0 + z_arr)) 
        dc_mpc = R0_mpc * np.log(np.tanh(t0_Tcurv / 2.0) / np.tanh(tz_Tcurv / 2.0))
        return np.abs(R0_mpc * np.sin(dc_mpc / R0_mpc))

T_MeV_grid = np.geomspace(1.0, 1e-3, 1000)
T_fine_grid = np.geomspace(0.0005e-6, 1.0e-5, 20000) 
lnT_fine_grid = np.log(T_fine_grid)

onion_bbn_file = max(glob.glob("bbn_Onion*.csv"), key=os.path.getmtime)
print(f"Reading base Onion abundances from: {onion_bbn_file}")

df_bbn = pd.read_csv(onion_bbn_file)
yp_onion = 4.0 * df_bbn['4He'].iloc[-1]
yd_onion = 2.0 * df_bbn['d'].iloc[-1]
y3he_onion = 3.0 * df_bbn['3He'].iloc[-1]
xp_onion = 1.0 * df_bbn['p'].iloc[-1]
T_nu_onion = df_bbn['T_nu'].iloc[-1] if 'T_nu' in df_bbn.columns else 0.0023

match_eta = re.search(r'_eta([0-9\.eE\+\-]+)_', onion_bbn_file)
eta_onion = float(match_eta.group(1)) if match_eta else 7.005e-9 
match_om = re.search(r'_Oml0([0-9\.]+)_', onion_bbn_file)
om_l0_onion = float(match_om.group(1)) if match_om else 0.52 
gs0_onion, gss0_onion = calc_g_star_limits(T_nu_onion)

print(f"LCDM Reference: eta={ETA_LCDM:.2e}, Yp={YP_LCDM:.4f}")
print(f"Onion Model: eta={eta_onion:.2e}, Yp={yp_onion:.4f}, OmegaL0={om_l0_onion}, T_nu={T_nu_onion:.4f}")

t0_sec = (1.0 / (H0_S_ONION * np.sqrt(om_l0_onion))) * np.arctanh(np.sqrt(om_l0_onion))
t0_yr = t0_sec / (3600 * 24 * 365.25)
print(f"Present-day age (Onion): {t0_yr/1e9:.3f} Gyr")

h_func_lcdm = lambda lnT: h_lcdm(lnT)
h_func_onion = lambda lnT: h_onion(lnT, Omega_L0=om_l0_onion)

interp_xe_lcdm = solve_peebles_recombination(ETA_LCDM, XP_LCDM, YP_LCDM, YD_LCDM, Y3HE_LCDM, h_func_lcdm)
T_tau1_lcdm, T_cmb_lcdm, tau_lcdm = compute_cmb_dynamics(T_fine_grid, lnT_fine_grid, interp_xe_lcdm, ETA_LCDM, XP_LCDM, YP_LCDM, YD_LCDM, Y3HE_LCDM, h_func_lcdm)
z_cmb_lcdm = (T_cmb_lcdm / T0) - 1.0
t_cmb_lcdm_yr = compute_event_time(T_cmb_lcdm, h_func_lcdm) / (3600 * 24 * 365.25)

rs_lcdm = compute_sound_horizon(T_cmb_lcdm, ETA_LCDM, h_func_lcdm, gss0=3.938)
chi_lcdm = compute_comoving_distance(T_cmb_lcdm, h_func_lcdm, gss0=3.938)
dm_lcdm = chi_lcdm
da_lcdm = dm_lcdm / (1.0 + z_cmb_lcdm)
theta_rad_lcdm = (rs_lcdm / (1.0 + z_cmb_lcdm)) / da_lcdm

print("\n--- LCDM RESULTS ---")
print(f"CMB Peak: z = {z_cmb_lcdm:.0f} | t = {t_cmb_lcdm_yr:,.0f} yr")
print(f"Sound horizon (rs): {rs_lcdm:,.2f} Mpc")
print(f"Comoving distance (DM): {dm_lcdm:,.2f} Mpc")
print(f"Angular scale (theta): {theta_rad_lcdm:.5f} rad")

interp_xe_onion = solve_peebles_recombination(eta_onion, xp_onion, yp_onion, yd_onion, y3he_onion, h_func_onion)
T_tau1_onion, T_cmb_onion, tau_onion = compute_cmb_dynamics(T_fine_grid, lnT_fine_grid, interp_xe_onion, eta_onion, xp_onion, yp_onion, yd_onion, y3he_onion, h_func_onion)
z_cmb_onion = (T_cmb_onion / T0) - 1.0
t_cmb_onion_yr = compute_event_time(T_cmb_onion, h_func_onion) / (3600 * 24 * 365.25)

chi_onion = compute_comoving_distance(T_cmb_onion, h_func_onion, gss0=gss0_onion)
dm_onion = compute_dm_onion(chi_onion, om_l0_onion)
da_onion = dm_onion / (1.0 + z_cmb_onion)
theta_rad_onion = (rs_lcdm / (1.0 + z_cmb_onion)) / da_onion

print("\n--- ONION RESULTS ---")
print(f"CMB Peak: z = {z_cmb_onion:.0f} | t = {t_cmb_onion_yr:,.0f} yr")
print(f"Comoving distance (DM): {dm_onion:,.2f} Mpc")
print(f"Angular scale (theta): {theta_rad_onion:.5f} rad")

if PLOT_EXPANSION_RATES:
    print("\nGenerating Expansion Rate Plots...")
    T_max = 1.22e21
    T_full = np.geomspace(T_max, T0, 2000)
    tlcdm_full, tonion_full = integrate_time(T_full, h_func_lcdm, h_func_onion)
    tonion_analytic_full = onion_time_analytic(T_full, Omega_L0=om_l0_onion)

    sec_per_yr = 3600 * 24 * 365.25
    t_lcdm_full_yr = tlcdm_full / sec_per_yr
    t_onion_full_yr = tonion_full / sec_per_yr
    t_onion_an_full_yr = tonion_analytic_full / sec_per_yr
    a_full = T0 / T_full

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(T_full, t_onion_an_full_yr, label='Onion (Analytic)', color='#FFFF99', linestyle='-', linewidth=6)
    ax1.plot(T_full, t_onion_full_yr, label='Onion (Integration)', color='red', linestyle='--', linewidth=1.5)
    ax1.plot(T_full, t_lcdm_full_yr, label=r'$\Lambda$CDM (Integration)', color='blue', linestyle='-', linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    yticks = [1e10, 1, 1e-10, 1e-20, 1e-30, 1e-40, 1e-50]
    ax1.set_yticks(yticks)
    ax1.invert_xaxis() 
    ax1.set_xlabel('Temperature $T$ (MeV)', fontsize=16)
    ax1.set_ylabel('Time $t$ (yr)', fontsize=16)
    ax1.set_title('Age-temperature relationship', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(loc='upper left', frameon=True, edgecolor='black', facecolor='white', fontsize=12)
    ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)

    ax2.plot(t_onion_an_full_yr, a_full, label='Onion (Analytic)', color='#FFFF99', linestyle='-', linewidth=6)
    ax2.plot(t_onion_full_yr, a_full, label='Onion (Integration)', color='red', linestyle='--', linewidth=1.5)
    ax2.plot(t_lcdm_full_yr, a_full, label=r'$\Lambda$CDM (Integration)', color='blue', linestyle='-', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time $t$ (yr)', fontsize=16)
    ax2.set_ylabel('Scale Factor $a$', fontsize=16)
    ax2.set_title('Age-scale factor relationship', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(loc='lower right', frameon=True, edgecolor='black', facecolor='white', fontsize=16)
    ax2.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('time_temperature.pdf', format='pdf', bbox_inches='tight')

if PLOT_Z_CMB:
    print("\nGenerating Z_CMB vs Eta Plot...")
    
    lcdm_sweep_file = max([f for f in glob.glob("abundances_sweep_*.csv") if 'LCDM' in f.upper()], key=os.path.getmtime)
    onion_sweep_file = max([f for f in glob.glob("abundances_sweep_*.csv") if 'LCDM' not in f.upper()], key=os.path.getmtime)
    print(f"Loading LCDM sweep from: {lcdm_sweep_file}")
    print(f"Loading Onion sweep from: {onion_sweep_file}")
    
    df_eta_lcdm = pd.read_csv(lcdm_sweep_file).sort_values(by='eta').reset_index(drop=True)
    df_eta_onion = pd.read_csv(onion_sweep_file).sort_values(by='eta').reset_index(drop=True)
    
    eta_lcdm_plot_list, z_cmb_lcdm_plot_list = [], []
    for _, row in df_eta_lcdm.iterrows():
        eta_test = row['eta']
        x_p_test = row['p'] 
        yp_test = 4.0 * row['4He']
        yd_test = 2.0 * row['d']
        y3he_test = 3.0 * row['3He']

        interp_xe_test = solve_peebles_recombination(eta_test, x_p_test, yp_test, yd_test, y3he_test, h_func_lcdm)
        _, t_cmb_test, _ = compute_cmb_dynamics(t_fine_grid, ln_t_fine_grid, interp_xe_test, eta_test, x_p_test, yp_test, yd_test, y3he_test, h_func_lcdm)
        
        eta_lcdm_plot_list.append(eta_test)
        z_cmb_lcdm_plot_list.append((t_cmb_test / T0) - 1.0)
        
    omega_search = 0.9706
    df_eta_onion_filtered = df_eta_onion[df_eta_onion['Omega_L0'] == omega_search].reset_index(drop=True)

    eta_onion_plot_list, z_cmb_onion_plot_list = [], []
    for _, row in df_eta_onion_filtered.iterrows():
        eta_test = row['eta']
        x_p_test = row['p'] 
        yp_test = 4.0 * row['4He']
        yd_test = 2.0 * row['d']
        y3he_test = 3.0 * row['3He']
        
        h_func_test = lambda lnT: h_onion(lnT, Omega_L0=omega_search)

        interp_xe_test = solve_peebles_recombination(eta_test, x_p_test, yp_test, yd_test, y3he_test, h_func_test)
        _, t_cmb_test, _ = compute_cmb_dynamics(t_fine_grid, ln_t_fine_grid, interp_xe_test, eta_test, x_p_test, yp_test, yd_test, y3he_test, h_func_test)
        
        eta_onion_plot_list.append(eta_test)
        z_cmb_onion_plot_list.append((t_cmb_test / T0) - 1.0)

    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(eta_onion_plot_list, z_cmb_onion_plot_list, color='darkorange', lw=3, label=fr'$z_{{CMB}}$ (Onion, $\Omega_{{\Lambda 0}} = {omega_search:.2f}$)')
    plt.plot(eta_lcdm_plot_list, z_cmb_lcdm_plot_list, color='royalblue', lw=3, label=r'$z_{CMB}$ ($\Lambda$CDM)')

    eta_fit = np.logspace(-10, -9, 1000)
    omega_b = (eta_fit * 1e10) / 274.0
    omega_m = 0.143
    g1 = (0.0783 * omega_b**-0.238) / (1.0 + 39.5 * omega_b**0.763)
    g2 = 0.560 / (1.0 + 21.1 * omega_b**1.81)
    z_star = 1048 * (1.0 + 0.00124 * omega_b**-0.738) * (1.0 + g1 * omega_m**g2)
    
    plt.plot(eta_fit, z_star, color='mediumseagreen', linestyle='-.', lw=2.5, label=r'$z_{CMB}$ fit (Hu & Sugiyama 1996)')
    plt.axhline(z_cmb_lcdm, color='gray', linestyle='--', lw=2, alpha=0.7, label=fr'$z_{{CMB}}$ LCDM ($\eta = {ETA_LCDM}$) $\approx$ {z_cmb_lcdm:.0f}')
    
    plt.xscale('log')
    plt.grid(True, which="both", alpha=0.4, linestyle='--')

    plt.xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
    plt.xlim(1e-10, 1e-6)
    plt.ylabel(r'$z_{CMB}$', fontsize=25)
    plt.legend(loc='best', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20) 

    plt.tight_layout()
    base_name_onion = os.path.splitext(os.path.basename(onion_sweep_file))[0]
    out_file = f"zcmb_vs_eta_{base_name_onion}.pdf"
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved: {out_file}")

if PLOT_TAU_XE:
    print("Generating Optical Depth and Ionization plots...")
    xe_lcdm_plot = interp_xe_lcdm(T_fine_grid)
    xe_onion_plot = interp_xe_onion(T_fine_grid)
    z_grid = (T_fine_grid / T0) - 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=200)

    ax1.plot(z_grid, tau_lcdm, label=r'$\Lambda$CDM', color='royalblue', lw=2.5)
    ax1.plot(z_grid, tau_onion, label=fr'Onion ($\Omega_{{\Lambda 0}} = {om_l0_onion:.2f}$, $\eta = {eta_onion:.2e}$)', color='tomato', lw=2.5)

    ax1.axhline(1.0, color='black', linestyle='--', lw=1.5, alpha=0.8, label=r'$\tau = 1$ (CMB)')
    ax1.axvline(z_cmb_lcdm, color='royalblue', linestyle=':', lw=1.5, alpha=0.6)
    ax1.axvline(z_cmb_onion, color='tomato', linestyle=':', lw=1.5, alpha=0.6)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", alpha=0.4, linestyle='--')

    ax1.set_xlabel(r'Redshift $z$', fontsize=30)
    ax1.set_ylabel(r'Optical depth $\tau$', fontsize=30)
    ax1.legend(loc='lower left', fontsize=23)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.invert_xaxis()

    ax2.plot(z_grid, xe_lcdm_plot, label=r'$\Lambda$CDM', color='royalblue', lw=2.5)
    ax2.plot(z_grid, xe_onion_plot, label=fr'Onion ($\Omega_{{\Lambda 0}} = {om_l0_onion:.2f}$, $\eta = {eta_onion:.2e}$)', color='tomato', lw=2.5)

    ax2.axvline(z_cmb_lcdm, color='royalblue', linestyle=':', lw=1.5, alpha=0.6)
    ax2.axvline(z_cmb_onion, color='tomato', linestyle=':', lw=1.5, alpha=0.6)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", alpha=0.4, linestyle='--')

    ax2.set_xlabel(r'Redshift $z$', fontsize=30)
    ax2.set_ylabel(r'Ionization fraction $X_e$', fontsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.invert_xaxis()

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2) 
    plt.savefig('tau_xe_dynamics.pdf', bbox_inches='tight')

if PLOT_DISTANCES:
    print("\nGenerating Distances vs Redshift Plot...")
    
    omega_values_to_plot = [0.0, 0.52, 0.85, om_l0_onion, 0.9999] 
    omega_values_to_plot = sorted(list(set(omega_values_to_plot))) 

    z_arr = np.geomspace(0.01, 1500, 1000)
    z_dots = np.geomspace(0.01, 1500, 25)

    t_arr = T0 * (1.0 + z_arr)

    chi_arr_lcdm = np.array([compute_comoving_distance(t_val, h_func_lcdm, gss0=3.938) for t_val in t_arr])
    dm_arr_lcdm = chi_arr_lcdm  
    da_arr_lcdm = dm_arr_lcdm / (1.0 + z_arr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, dpi=300)

    ax1.plot(z_arr, dm_arr_lcdm/rs_lcdm, label=r'$\Lambda$CDM', color='blue', lw=3, zorder=10)
    ax2.plot(z_arr, da_arr_lcdm/rs_lcdm, label=r'$\Lambda$CDM', color='blue', lw=3, zorder=10)
    
    ax1.axvline(z_cmb_lcdm, color='blue', linestyle=':', lw=1.5, alpha=0.5)
    ax1.plot(z_cmb_lcdm, dm_lcdm/rs_lcdm, 'bo', markersize=8, markeredgecolor='black', zorder=15)
    
    ax2.axvline(z_cmb_lcdm, color='blue', linestyle=':', lw=1.5, alpha=0.5)
    ax2.plot(z_cmb_lcdm, da_lcdm/rs_lcdm, 'bo', markersize=8, markeredgecolor='black', zorder=15)

    colores_onion = cm.OrRd(np.linspace(0.3, 1.0, len(omega_values_to_plot)))

    for idx, om_val in enumerate(omega_values_to_plot):
        h_func_temp = lambda lnT, ov=om_val: h_onion(lnT, Omega_L0=ov)
        
        chi_arr_temp = np.array([compute_comoving_distance(t_val, h_func_temp, gss0=gss0_onion) for t_val in t_arr])
        dm_arr_temp = np.array([compute_dm_onion(chi, om_val) for chi in chi_arr_temp])
        da_arr_temp = dm_arr_temp / (1.0 + z_arr)
        
        dm_analitico = compute_dm_onion_analytic(z_dots, om_val)
        da_analitico = dm_analitico / (1.0 + z_dots)
        
        if np.isclose(om_val, om_l0_onion):
            grosor = 3.0
            estilo = '-'
            alfa = 1.0
            color_linea = 'red' 
            
            ax1.axvline(z_cmb_onion, color=color_linea, linestyle=':', lw=1.5, alpha=0.5)
            ax1.plot(z_cmb_onion, dm_onion/rs_lcdm, 'ro', markersize=8, markeredgecolor='black', zorder=15)
            ax2.axvline(z_cmb_onion, color=color_linea, linestyle=':', lw=1.5, alpha=0.5)
            ax2.plot(z_cmb_onion, da_onion/rs_lcdm, 'ro', markersize=8, markeredgecolor='black', zorder=15)
            
            label_analitico = 'Onion analytical'
        else:
            grosor = 1.5
            estilo = '--'
            alfa = 0.7
            color_linea = colores_onion[idx]
            label_analitico = None

        ax1.plot(z_arr, dm_arr_temp/rs_lcdm, label=fr'Onion ($\Omega_{{\Lambda 0}}={om_val:.2f}$)', 
                 color=color_linea, lw=grosor, linestyle=estilo, alpha=alfa)
        ax2.plot(z_arr, da_arr_temp/rs_lcdm, color=color_linea, lw=grosor, linestyle=estilo, alpha=alfa)
        
        ax1.plot(z_dots, dm_analitico/rs_lcdm, marker='o', markersize=8, color=color_linea, 
                 linestyle='none', alpha=alfa, label=label_analitico)
        ax2.plot(z_dots, da_analitico/rs_lcdm, marker='o', markersize=8, color=color_linea, 
                 linestyle='none', alpha=alfa)

    ax1.set_ylabel(r'Comoving distance $D_M/r_s$', fontsize=25)
    ax1.grid(True, which="both", alpha=0.4, linestyle='--')
    ax1.set_ylim(0.0, 300)
    ax1.legend(loc='upper right', fontsize=18, frameon=True) 

    ax2.set_xlabel(r'Redshift $z$', fontsize=25)
    ax2.set_ylabel(r'Angular distance $D_A/r_s$', fontsize=25)
    ax2.grid(True, which="both", alpha=0.4, linestyle='--')
    
    ax2.set_xscale('log')
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_xticks([1, 10, 100, 1000]) 

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    ax2.invert_xaxis() 
        
    out_file = "distancias_multiples.pdf"
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved: {out_file}")

if COMPUTE_2D_GRID:
    onion_sweep_file = max([f for f in glob.glob("abundances_sweep_*.csv") if 'LCDM' not in f.upper()], key=os.path.getmtime)
    df_onion_grid = pd.read_csv(onion_sweep_file)
    print(f"\nComputing 2D physical grid from {onion_sweep_file}...")
    
    eta_lst, om_list, yp_lst, dh_lst, th_lst = [], [], [], [], []
    total_pts = len(df_onion_grid)
    start_time = time.time()
    
    for i, row in df_onion_grid.iterrows():
        eta_val = row['eta']
        omega_val = row['Omega_L0']
        t_nu_val = row['T_nu']
        yp_val = 4.0 * row['4He']
        dh_val = row['d'] / row['p']
        x_p_val = 1.0 * row['p']
        yd_val = 2.0 * row['d']
        y3he_val = 3.0 * row['3He']
        
        _, gss0_test = calc_g_star_limits(t_nu_val)
        h_func_test = lambda lnt, ov=omega_val: h_onion(lnt, Omega_L0=ov)

        interp_xe_test = solve_peebles_recombination(eta_val, x_p_val, yp_val, yd_val, y3he_val, h_func_test)
        _, t_cmb_test, _ = compute_cmb_dynamics(t_fine_grid, ln_t_fine_grid, interp_xe_test, eta_val, x_p_val, yp_val, yd_val, y3he_val, h_func_test)
        
        z_cmb_test = (t_cmb_test / T0) - 1.0
        chi_test = compute_comoving_distance(t_cmb_test, h_func_test, gss0=gss0_test)
        dm_test = compute_dm_onion(chi_test, omega_val)
        
        theta_test = (RS_PLANCK / (1.0 + z_cmb_test)) / (dm_test / (1.0 + z_cmb_test))
        
        eta_lst.append(eta_val)
        om_list.append(omega_val)
        yp_lst.append(yp_val)
        dh_lst.append(dh_val)
        th_lst.append(theta_test)
        
        if (i + 1) % 10 == 0 or (i + 1) == total_pts:
            elap = time.time() - start_time
            rem = (elap / (i + 1)) * (total_pts - (i + 1))
            print(f"Progress: {i+1}/{total_pts} | Est. remaining time: {rem/60:.1f} min", end='\r')

    print() 
    
    df_2d_physical = pd.DataFrame({
        'eta': eta_lst,
        'Omega_L0': om_list,
        'Yp': yp_lst,
        'D_H': dh_lst,
        'theta': th_lst
    })
    
    base_name = os.path.splitext(os.path.basename(onion_sweep_file))[0]
    out_csv = f"physical_grid_2D_{base_name}.csv"
    df_2d_physical.to_csv(out_csv, index=False)
    print(f"Saved 2D grid to {out_csv}")

if PLOT_CONTOURS:
    grid_files = glob.glob("physical_grid_2D_*.csv")
    
    if not grid_files:
        print("Warning: No 2D grid files found. Skipping contour plots.")
    else:
        grid_file = max(grid_files, key=os.path.getmtime)
        print(f"\nPlotting contours from: {grid_file}")
        
        df_grid = pd.read_csv(grid_file)
        
        eta_arr = df_grid['eta'].values
        om_arr = df_grid['Omega_L0'].values
        yp_arr = df_grid['Yp'].values
        dh_arr = df_grid['D_H'].values
        th_arr = df_grid['theta'].values

        unique_oms = np.sort(np.unique(om_arr))
        eta_log = np.log10(eta_arr)
        eta_dense_log = np.linspace(eta_log.min(), eta_log.max(), 1000)

        yp_2d = np.zeros((len(unique_oms), len(eta_dense_log)))
        dh_2d = np.zeros((len(unique_oms), len(eta_dense_log))) 
        th_2d = np.zeros((len(unique_oms), len(eta_dense_log)))

        for idx, om in enumerate(unique_oms):
            mask = (om_arr == om)
            e_log, y, d, t = eta_log[mask], yp_arr[mask], dh_arr[mask], th_arr[mask]
            
            srt = np.argsort(e_log)
            e_log, y, d, t = e_log[srt], y[srt], d[srt], t[srt]
            
            yp_2d[idx, :] = interp1d(e_log, y, kind='cubic', fill_value="extrapolate")(eta_dense_log)
            dh_2d[idx, :] = interp1d(e_log, d, kind='cubic', fill_value="extrapolate")(eta_dense_log)
            
            valid_mask = ~np.isnan(t)
            if np.sum(valid_mask) > 3:
                th_2d[idx, :] = interp1d(e_log[valid_mask], t[valid_mask], kind='cubic', fill_value="extrapolate")(eta_dense_log)
            else:
                th_2d[idx, :] = np.nan

        th_2d_100 = th_2d * 100.0
        
        theta_obs_100, sigma_theta_100 = 1.04105, 0.00046
        yp_obs, sigma_yp = 0.245, 0.003
        dh_obs, sigma_dh = 2.547e-5, 0.025e-5

        lik_yp = np.exp(-0.5 * ((yp_2d - yp_obs) / sigma_yp)**2)
        lik_d = np.exp(-0.5 * ((dh_2d - dh_obs) / sigma_dh)**2)
        
        lik_th = np.zeros_like(th_2d_100)
        valid_th_dense = ~np.isnan(th_2d_100)
        lik_th[valid_th_dense] = np.exp(-0.5 * ((th_2d_100[valid_th_dense] - theta_obs_100) / sigma_theta_100)**2)

        eta_mesh_real = 10**np.meshgrid(eta_dense_log, unique_oms)[0]
        om_mesh = np.meshgrid(eta_dense_log, unique_oms)[1]

        def get_contours(grid):
            flat = grid[~np.isnan(grid)]
            if np.sum(flat) == 0: 
                return [1.0, 1.0, 1.0]
            srt = np.sort(flat)[::-1]
            cum = np.cumsum(srt / np.sum(srt))
            
            lvl_3sig = srt[np.searchsorted(cum, 0.9973)] if np.searchsorted(cum, 0.9973) < len(srt) else srt[-1]
            lvl_2sig = srt[np.searchsorted(cum, 0.9545)] if np.searchsorted(cum, 0.9545) < len(srt) else srt[-1]
            lvl_1sig = srt[np.searchsorted(cum, 0.6827)] if np.searchsorted(cum, 0.6827) < len(srt) else srt[-1]
            
            return [lvl_3sig, lvl_2sig, lvl_1sig]

        c_yp = get_contours(lik_yp)
        c_d = get_contours(lik_d)
        c_th = get_contours(lik_th)

        line_1sig = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1.5, label=r'$1\sigma$')
        line_2sig = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label=r'$2\sigma$')
        line_3sig = mlines.Line2D([], [], color='black', linestyle=':', linewidth=1.5, label=r'$3\sigma$')

        fig_d, ax_d = plt.subplots(1, 1, figsize=(10, 5), dpi=200)

        cf_d = ax_d.contourf(eta_mesh_real, om_mesh, dh_2d, levels=100, cmap='cividis')
        if c_d[0] < 1.0:
            ax_d.contour(eta_mesh_real, om_mesh, lik_d, levels=c_d, 
                         colors=['black', 'black', 'black'], linestyles=[':', '--', '-'], 
                         linewidths=[0.8, 1.2, 1.5], alpha=0.9)
            
        cb_d = fig_d.colorbar(cf_d, ax=ax_d, pad=0.02)
        cb_d.set_label(r'$D/H$', fontsize=24)
        cb_d.ax.tick_params(labelsize=16)

        ax_d.set_xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
        ax_d.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
        ax_d.grid(True, alpha=0.3, ls='--')
        ax_d.set_xscale('log')
        ax_d.tick_params(axis='both', which='major', labelsize=16)
        ax_d.tick_params(axis='x', which='minor', labelsize=16)
        ax_d.legend(handles=[line_1sig, line_2sig, line_3sig], loc='upper right', fontsize=20)

        fig_d.tight_layout()
        plt.savefig("contour_DH.pdf", format='pdf', bbox_inches='tight')
        print("Solo D/H contour plot saved as 'contour_DH.pdf'")

        fig, (ax_yp, ax_th) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, dpi=200)
        
        cf_yp = ax_yp.contourf(eta_mesh_real, om_mesh, yp_2d, levels=100, cmap='plasma')
        if c_yp[0] < 1.0:
            ax_yp.contour(eta_mesh_real, om_mesh, lik_yp, levels=c_yp, 
                          colors=['black', 'black', 'black'], linestyles=[':', '--', '-'], 
                          linewidths=[0.8, 1.2, 1.5], alpha=1.0)
            
        cb_yp = fig.colorbar(cf_yp, ax=ax_yp, pad=0.02)
        cb_yp.set_label(r'$Y_p$', fontsize=24)
        cb_yp.ax.tick_params(labelsize=16)
        ax_yp.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
        ax_yp.grid(True, alpha=0.3, ls='--')
        ax_yp.legend(handles=[line_1sig, line_2sig, line_3sig], loc='upper right', fontsize=20)
        ax_yp.tick_params(axis='both', which='major', labelsize=16)

        levels_th = np.linspace(0.0, 2.0, 100)
        cf_th = ax_th.contourf(eta_mesh_real, om_mesh, th_2d_100, levels=levels_th, cmap='viridis', extend='max')
        
        exact_levels = [theta_obs_100 + mult * sigma_theta_100 for mult in [-3, -2, -1, 1, 2, 3]]
        ax_th.contour(eta_mesh_real, om_mesh, th_2d_100, levels=exact_levels, 
                      colors='black', linestyles=[':', '--', '-', '-', '--', ':'], 
                      linewidths=[1.5, 1.5, 2, 2, 1.5, 1.5], alpha=0.8)
                      
        cb_th = fig.colorbar(cf_th, ax=ax_th, pad=0.02)
        cb_th.set_label(r'$100\theta_{CMB}$', fontsize=24)
        cb_th.ax.tick_params(labelsize=16)

        ax_th.set_xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
        ax_th.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
        ax_th.grid(True, alpha=0.3, ls='--')
        
        line_overlap = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2.5, label=r'$1\sigma, 2\sigma, 3\sigma$')
        ax_th.legend(handles=[line_overlap], loc='upper right', fontsize=18)
        
        ax_th.set_xscale('log')
        ax_th.tick_params(axis='both', which='major', labelsize=16)
        ax_th.tick_params(axis='x', which='minor', labelsize=16)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.08)
        plt.savefig("contour_Yp_and_theta_combined.pdf", format='pdf', bbox_inches='tight')
        print("Combined Yp/Theta plot saved as 'contour_Yp_and_theta_combined.pdf'")
        
        om_dense = np.linspace(unique_oms.min(), unique_oms.max(), 5000)
        
        yp_sup = interp1d(unique_oms, yp_2d, axis=0)(om_dense)
        th_sup = interp1d(unique_oms, th_2d_100, axis=0)(om_dense)
        
        eta_super, om_super = np.meshgrid(eta_dense_log, om_dense)
        eta_super_real = 10**eta_super
        
        mask_yp = np.abs(yp_sup - yp_obs) <= 2 * sigma_yp
        mask_th = np.abs(th_sup - theta_obs_100) <= 2 * sigma_theta_100
        intersection = mask_yp & mask_th
        
        print("\n--- 2-Sigma Intersection Analysis (Yp & Theta_CMB) ---")
        if np.any(intersection):
            dilated = ndimage.binary_dilation(intersection, structure=np.ones((20, 20)))
            labeled, num_features = ndimage.label(dilated)
            print(f"Found {num_features} main intersection region(s):")
            
            for i in range(1, num_features + 1):
                region = (labeled == i) & intersection
                if not np.any(region): 
                    continue
                    
                e_min, e_max = eta_super_real[region].min(), eta_super_real[region].max()
                o_min, o_max = om_super[region].min(), om_super[region].max()
                
                eta_c, eta_err = (e_max + e_min) / 2.0, (e_max - e_min) / 2.0
                om_c, om_err = (o_max + o_min) / 2.0, (o_max - o_min) / 2.0
                
                exp_eta = int(np.floor(np.log10(eta_c)))
                
                print(f"\n Region {i}:")
                print(f"  eta      = ({eta_c / 10**exp_eta:.3f} +/- {eta_err / 10**exp_eta:.3f}) x 10^{exp_eta}")
                print(f"  Omega_L0 = {om_c:.7f} +/- {om_err:.7f}")
        else:
            print("No 2-sigma intersection found between Yp and Theta_CMB.")
plt.show()
