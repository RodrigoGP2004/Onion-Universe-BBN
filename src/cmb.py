#%% IMPORTACIÓN DE VARIABLES Y FUNCIONES A USAR
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import brentq, minimize
import matplotlib.lines as mlines
import matplotlib.lines as mlines
from scipy.interpolate import interp1d
from scipy.special import zeta
import time
import pandas as pd
import glob
import os
import sys
import scipy.ndimage as ndimage
import re
import matplotlib.cm as cm

directorio_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(directorio_script)
print(f"Directorio de trabajo fijado correctamente en: {os.getcwd()}")

Qnp = 1.293332       
tau_n = 879.4     
m_e = 0.51099895       
T0 = 2.348e-10               
B_H = 13.6e-6
B_He = 24.5876e-6                 
sigma_T_cm2 = 6.6524587e-25
c_light = 2.99792458e10 #cm/s
lambda_alpha_cm = 1.215682e-5 
Lambda_2s1s = 8.224           
km_to_m = 1e3
Mpc_to_m = 3.08567758e22
hbar_c_MeV_cm = 1.9732698e-11 
m_p_MeV = 938.272

g_star_i = 10.75
T_ee = 0.187

H0_km_s_Mpc_LCDM = 71.3 
H0_km_s_Mpc_Onion = 71.3 
H0_s_LCDM = H0_km_s_Mpc_LCDM * km_to_m / Mpc_to_m
H0_s_Onion = H0_km_s_Mpc_Onion * km_to_m / Mpc_to_m

eta_LCDM = 6.1395e-10
Yp_LCDM = 0.2462715
YD_LCDM = 3.808996e-05
Y3He_LCDM = 2.424953e-05   
X_p_LCDM = 1.0 - Yp_LCDM - YD_LCDM - Y3He_LCDM

rs_Planck = 147

def calc_g_star_limits(T_nu):
    if T_nu is None or np.isnan(T_nu): return 3.384, 3.938
    x = 3.0 * np.log(T_nu / 0.18)
    W = 0.5 * (1.0 + np.tanh(x))
    gs0 = W * 3.384 + (1.0 - W) * 7.25
    gss0 = W * 3.938 + (1.0 - W) * 7.25
    return gs0, gss0

def get_g_star(T_MeV, gs0=3.384):
    x = 3.0 * np.log(T_MeV / 0.18)
    return gs0 + (g_star_i - gs0) * 0.5 * (1.0 + np.tanh(x))

def get_g_star_s(T_MeV, gss0=3.938):
    x = 3.0 * np.log(T_MeV / 0.18)
    return gss0 + (g_star_i - gss0) * 0.5 * (1.0 + np.tanh(x))

def dln_g_dln_T(T_MeV, gss0=3.938):
    x = 3.0 * np.log(T_MeV / 0.18)
    th = np.tanh(x)
    dg_dlnT = (g_star_i - gss0) * 0.5 * 3.0 * (1.0 - th**2)
    g_s = get_g_star_s(T_MeV, gss0)
    return dg_dlnT / g_s

def H_LCDM(lnT, Omega_L0=7186):
    T = np.exp(lnT)
    h = H0_km_s_Mpc_LCDM / 100.0 
    omega_r = 2.473e-5 + 1.710e-5
    omega_m = 0.143
    
    Omega_r0 = omega_r / (h**2) 
    Omega_m0 = omega_m / (h**2)
    Omega_k0 = 0.0 
    Omega_L0 = 1.0 - (Omega_r0 + Omega_m0)
    
    ratio = T / T0
    return H0_s_LCDM * np.sqrt(Omega_r0 * (get_g_star(T, 3.384) / 3.384) * (ratio**4) + Omega_m0 * (get_g_star_s(T, 3.938) / 3.938) * (ratio**3) + Omega_k0 * (ratio**2) + Omega_L0)

def H_Onion(lnT, Omega_L0=0.52):
    T = np.exp(lnT)
    Omega_k0 = Omega_L0 - 1.0
    return H0_s_Onion * np.sqrt(-Omega_k0 * pow(T/T0, 2) + Omega_L0)

def tiempo_integracion(Temp, H_func1, H_func2):
    lnT_inf = np.log(5e21) 
    tiempo_1, tiempo_2 = [],[]
    for T_val in Temp:
        lnT = np.log(T_val)
        t1, _ = quad(lambda x: (1.0 + dln_g_dln_T(np.exp(x))/3.0) / H_func1(x), lnT, lnT_inf)
        t2, _ = quad(lambda x: (1.0 + dln_g_dln_T(np.exp(x))/3.0) / H_func2(x), lnT, lnT_inf)
        tiempo_1.append(t1)
        tiempo_2.append(t2)
    return np.array(tiempo_1), np.array(tiempo_2)

def Tiempo_Onion_Analitico(Temp, Omega_L0=0.97):
    TCurv = 1.0 / (H0_s_Onion * np.sqrt(Omega_L0))
    t0 = 0.5 * TCurv * np.log((1.0 + np.sqrt(Omega_L0)) / (1.0 - np.sqrt(Omega_L0)))
    return TCurv * np.arcsinh(T0 * np.sinh(t0/TCurv)/Temp)

def calcular_tiempo_evento(T_target, H_func):
    lnT_rec = np.log(T_target)
    lnT_inf = np.log(5e21)
    t, _ = quad(lambda x: (1.0 + dln_g_dln_T(np.exp(x))/3.0) / H_func(x), lnT_rec, lnT_inf)
    return t

def fraccion_ionizacion_He_saha(T_array, eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val):
    frac_num_H_tot = X_p_val + (X_D_val / 2.0)
    frac_num_He_tot = (X_4He_val / 4.0) + (X_3He_val / 3.0)
    if frac_num_He_tot < 1e-15: return np.zeros_like(T_array)
    f_H = frac_num_H_tot / frac_num_He_tot 
    coef = 4.0 * (np.pi**2) / (2.0 * zeta(3) * eta_val * frac_num_He_tot)
    term_masa = (m_e / (2.0 * np.pi * T_array))**1.5
    with np.errstate(over='ignore', under='ignore'):
        S_T = coef * term_masa * np.exp(-B_He / T_array)
    b = f_H + S_T
    x_He = (-b + np.sqrt(b**2 + 4.0 * S_T)) / 2.0 
    return np.clip(np.nan_to_num(x_He, nan=0.0), 0.0, 1.0)

def fraccion_ionizacion_Xe_saha(T_MeV, eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val):
    frac_num_H_tot = X_p_val + (X_D_val / 2.0)
    if frac_num_H_tot < 1e-15: return 1.0 
    coef = (np.pi**2) / (2.0 * zeta(3) * eta_val * frac_num_H_tot)
    term_masa = (m_e / (2.0 * np.pi * T_MeV))**1.5
    S_T = coef * term_masa * np.exp(-B_H / T_MeV)
    return (-S_T + np.sqrt(S_T**2 + 4.0*S_T)) / 2.0

def peebles_dXedx(x, X_e_array, eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val, H_func):
    X_e = max(1e-15, min(1.0, X_e_array[0]))
    T_MeV = B_H / x
    H_s = H_func(np.log(T_MeV))
    
    n_gamma = (2.0 * zeta(3) / np.pi**2) * (T_MeV / hbar_c_MeV_cm)**3
    n_H_tot = eta_val * n_gamma * (X_p_val + (X_D_val / 2.0))
    if n_H_tot < 1e-30: return [0.0]
    
    # Coeficiente alpha_B de RECFAST (Pequignot et al. 1991)
    T_K = T_MeV * 1.16045e10
    t4 = T_K / 10000.0
    alpha_B = 1.14 * 4.309e-13 * (t4**-0.6166) / (1.0 + 0.6703 * (t4**0.53))
    
    term_masa_cm3 = (m_e * T_MeV / (2.0 * np.pi))**1.5 / (hbar_c_MeV_cm**3)
    beta_B = alpha_B * term_masa_cm3 * np.exp(-x / 4.0)
    
    n_H_neutral = n_H_tot * max(1.0 - X_e, 1e-15) 
    R_Lya = 1e30 if n_H_neutral < 1e-30 else (8.0 * np.pi * H_s) / (3.0 * (lambda_alpha_cm**3) * n_H_neutral)
    
    rate_down_eff = 3.0 * R_Lya + Lambda_2s1s
    C_factor = rate_down_eff / (beta_B + rate_down_eff)
    dX_e_dt = - C_factor * (n_H_tot * (X_e**2) * alpha_B - (1.0 - X_e) * beta_B * np.exp(-0.75 * x))
    factor_tiempo = 1.0 + dln_g_dln_T(T_MeV) / 3.0
    return[(dX_e_dt * factor_tiempo) / (x * H_s)]

def resolver_recombinacion_peebles(eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val, H_func):
    T_start, T_end = 0.40e-6, 0.0005e-6  
    x_start, x_end = B_H / T_start, B_H / T_end
    X_e_start = fraccion_ionizacion_Xe_saha(T_start, eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val)
    sol = solve_ivp(peebles_dXedx, [x_start, x_end],[X_e_start], method='BDF',
                    t_eval=np.linspace(x_start, x_end, 100000), args=(eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val, H_func), 
                    rtol=1e-6, atol=1e-8)
    if not sol.success: return lambda T: np.ones_like(T) if isinstance(T, np.ndarray) else 1.0
    T_arr = B_H / np.asarray(sol.t)
    return interp1d(T_arr[::-1], sol.y[0][::-1], kind='linear', bounds_error=False, fill_value=(sol.y[0][-1], 1.0))

def calcular_dinamica_CMB(T_array, lnT_array, interp_Xe, eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val, H_func):
    H_s = np.array([H_func(lnt) for lnt in lnT_array])
    X_He = fraccion_ionizacion_He_saha(T_array, eta_val, X_p_val, X_4He_val, X_D_val, X_3He_val) 
    
    n_b = eta_val * (2.0 * zeta(3) / np.pi**2) * (T_array / hbar_c_MeV_cm)**3
    n_H_tot = n_b * (X_p_val + (X_D_val / 2.0))
    n_He_tot = n_b * ((X_4He_val / 4.0) + (X_3He_val / 3.0)) 
    
    n_e = np.maximum(n_H_tot * np.maximum(0.0, interp_Xe(T_array)) + n_He_tot * X_He, 1e-30)
    
    factor_tiempo = 1.0 + dln_g_dln_T(T_array) / 3.0
    dtau_dlnT = (c_light * sigma_T_cm2 * n_e * factor_tiempo) / H_s
    tau_array = cumulative_trapezoid(dtau_dlnT, lnT_array, initial=0)
    
    idx_tau1 = np.argmin(np.abs(tau_array - 1.0))
    idx_peak = np.argmax(dtau_dlnT * np.exp(-tau_array))
    return T_array[idx_tau1], T_array[idx_peak], tau_array

def calcular_horizonte_sonido(T_target, eta_val, H_func, gss0):
    def integrando_rs(lnT):
        T = np.exp(lnT)
        g_star_s_ratio = get_g_star_s(T, gss0) / gss0
        a_T = (T0 / T) * (1.0 / g_star_s_ratio)**(1.0/3.0) 
        
        n_gamma = (2.0 * zeta(3) / np.pi**2) * (T / hbar_c_MeV_cm)**3 
        rho_gamma = (np.pi**2 / 15.0) * (T**4) / (hbar_c_MeV_cm**3) 
        rho_b = eta_val * n_gamma * m_p_MeV * g_star_s_ratio 
        
        cs = c_light / np.sqrt(3.0 * (1.0 + (3.0 * rho_b) / (4.0 * rho_gamma))) 
        factor_tiempo = 1.0 + dln_g_dln_T(T, gss0) / 3.0
        return (cs / a_T) * factor_tiempo / H_func(lnT) 
    
    rs_cm, _ = quad(integrando_rs, np.log(T_target), np.log(5e21)) 
    return rs_cm / (Mpc_to_m * 100.0)

def calcular_distancia_comovil(T_target, H_func, gss0):
    def integrando_chi(lnT):
        T = np.exp(lnT)
        a_T = (T0 / T) * (1.0 / (get_g_star_s(T, gss0) / gss0))**(1.0/3.0)
        factor_tiempo = 1.0 + dln_g_dln_T(T, gss0) / 3.0
        return c_light * factor_tiempo / (a_T * H_func(lnT))
    
    chi_cm, _ = quad(integrando_chi, np.log(T0), np.log(T_target))
    return chi_cm / (Mpc_to_m * 100.0)

def calcular_DM_LCDM(chi_Mpc):
    return chi_Mpc

def calcular_DM_Onion(chi_Mpc, Omega_L0):
    # k=1 siempre por definición
    # D_M(t) = R_0 * sin(D_c(t) / R_0)
    
    if Omega_L0 < 1e-8:
        R0_cm = c_light / H0_s_Onion
    else:
        # Omega_L0 > 0 (k_st < 0)
        TCurv = 1.0 / (H0_s_Onion * np.sqrt(Omega_L0))
        t0_s = TCurv * np.arctanh(np.sqrt(Omega_L0))
        R0_cm = c_light * TCurv * np.sinh(t0_s / TCurv)
        
    R0_Mpc = R0_cm / (Mpc_to_m * 100.0)
    
    return np.abs(R0_Mpc * np.sin(chi_Mpc / R0_Mpc))

def calcular_DM_Onion_Analitico(z_array, Omega_L0):
    if Omega_L0 < 1e-8:
        R0_cm = c_light / H0_s_Onion
        R0_Mpc = R0_cm / (Mpc_to_m * 100.0)
        Dc_Mpc = R0_Mpc * np.log(1.0 + z_array)
        DM_Mpc = np.abs(R0_Mpc * np.sin(Dc_Mpc / R0_Mpc))
        return DM_Mpc
    else:
        # Caso k_st < 0
        TCurv = 1.0 / (H0_s_Onion * np.sqrt(Omega_L0))
        t0_TCurv = np.arctanh(np.sqrt(Omega_L0)) # t_0 / T
        R0_cm = c_light * TCurv * np.sinh(t0_TCurv)
        R0_Mpc = R0_cm / (Mpc_to_m * 100.0)
        
        # a(t) = sinh(t/T) / sinh(t0/T)
        tz_TCurv = np.arcsinh(np.sinh(t0_TCurv) / (1.0 + z_array)) # t(z) / TCurv
        
        val_t0 = np.tanh(t0_TCurv / 2.0)
        val_tz = np.tanh(tz_TCurv / 2.0)
        Dc_Mpc = R0_Mpc * np.log(val_t0 / val_tz)
        
        DM_Mpc = np.abs(R0_Mpc * np.sin(Dc_Mpc / R0_Mpc))
        return DM_Mpc

TMeV = np.geomspace(1.0, 1e-3, 1000)
T_grid = np.geomspace(0.0005e-6, 1.0e-5, 20000) 
lnT_grid = np.log(T_grid)

#%% CARGAR DATOS PARA UN CÓMPUTO DE ONION CONCRETO

archivos_bbn_Onion = glob.glob("resultados_bbn_Onion*.csv")
archivo_bbn_Onion_reciente = max(archivos_bbn_Onion, key=os.path.getmtime)
print(f"Leyendo abundancias base Onion desde: {archivo_bbn_Onion_reciente}")

df_bbn = pd.read_csv(archivo_bbn_Onion_reciente)
Yp_Onion = 4.0 * df_bbn['4He'].iloc[-1]
YD_Onion = 2.0 * df_bbn['d'].iloc[-1]
Y3He_Onion = 3.0 * df_bbn['3He'].iloc[-1]
X_p_Onion = 1.0 * df_bbn['p'].iloc[-1]
T_nu_Onion_base = df_bbn['T_nu'].iloc[-1] if 'T_nu' in df_bbn.columns else 0.0023

match_eta = re.search(r'_eta([0-9\.eE\+\-]+)_', archivo_bbn_Onion_reciente) #Lee el valor de eta desde el nombre del archivo en sí
eta_Onion = float(match_eta.group(1)) if match_eta else 7.005e-9 
match_om = re.search(r'_Oml0([0-9\.]+)_', archivo_bbn_Onion_reciente)
Oml0_Onion_base = float(match_om.group(1)) if match_om else 0.52 
gs0_Onion_base, gss0_Onion_base = calc_g_star_limits(T_nu_Onion_base)

print(f" -> LCDM Fijo: eta={eta_LCDM:.2e}, Yp={Yp_LCDM:.4f} (g*s0=3.938)")
print(f" -> Onion CSV: eta={eta_Onion:.2e}, Yp={Yp_Onion:.4f}, OmegaL0={Oml0_Onion_base}, T_nu={T_nu_Onion_base:.4f} (g*s0={gss0_Onion_base:.3f})")

sq_Oml0 = np.sqrt(Oml0_Onion_base)
t0_segundos = (1.0 / (H0_s_Onion * sq_Oml0)) * np.arctanh(sq_Oml0)
t0_anios = t0_segundos / (3600 * 24 * 365.25)

print(f" -> Edad actual Universo Onion: {t0_anios/1e9:.3f} mil millones de años ({t0_anios:.4e} años)")

#%% CÁLCULOS PARA LCDM BASE Y ONION BASE

H_func_LCDM = lambda lnT: H_LCDM(lnT)
H_func_Onion_base = lambda lnT: H_Onion(lnT, Omega_L0=Oml0_Onion_base)

tLCDM, tOnion = tiempo_integracion(TMeV, H_func_LCDM, H_func_Onion_base)
tOnion2 = Tiempo_Onion_Analitico(TMeV, Omega_L0=Oml0_Onion_base)

interp_Xe_LCDM = resolver_recombinacion_peebles(eta_LCDM, X_p_LCDM, Yp_LCDM, YD_LCDM, Y3He_LCDM, H_func_LCDM)
T_tau1_LCDM, T_cmb_LCDM, tau_LCDM = calcular_dinamica_CMB(T_grid, lnT_grid, interp_Xe_LCDM, eta_LCDM, X_p_LCDM, Yp_LCDM, YD_LCDM, Y3He_LCDM, H_func_LCDM)
z_cmb_LCDM = (T_cmb_LCDM / T0) - 1.0
t_cmb_LCDM_anios = calcular_tiempo_evento(T_cmb_LCDM, H_func_LCDM) / (3600 * 24 * 365.25)

rs_LCDM = calcular_horizonte_sonido(T_cmb_LCDM, eta_LCDM, H_func_LCDM, gss0=3.938)
chi_LCDM = calcular_distancia_comovil(T_cmb_LCDM, H_func_LCDM, gss0=3.938)
DM_LCDM = calcular_DM_LCDM(chi_LCDM)
DA_LCDM = DM_LCDM / (1.0 + z_cmb_LCDM)
theta_rad_LCDM = (rs_LCDM / (1.0 + z_cmb_LCDM)) / DA_LCDM

print("="*65)
print(" RESULTADOS LCDM ")
print("="*65)
print(f"Pico Visibilidad (CMB): z = {z_cmb_LCDM:.0f} | t = {t_cmb_LCDM_anios:,.0f} años")
print(f"Radio comóvil r_s: {rs_LCDM:,.2f} Mpc")
print(f"Dist. comóvil D_M: {DM_LCDM:,.2f} Mpc")
print(f"Dist. angular D_A: {DA_LCDM:,.2f} Mpc")
print(f"Ángulo theta: {theta_rad_LCDM:.5f} rad")

# CÁLCULOS PARA ONION BASE (ESPECÍFICO DEL ARCHIVO CARGADO)

interp_Xe_Onion = resolver_recombinacion_peebles(eta_Onion, X_p_Onion, Yp_Onion, YD_Onion, Y3He_Onion, H_func_Onion_base)
T_tau1_Onion, T_cmb_Onion, tau_Onion = calcular_dinamica_CMB(T_grid, lnT_grid, interp_Xe_Onion, eta_Onion, X_p_Onion, Yp_Onion, YD_Onion, Y3He_Onion, H_func_Onion_base)
z_cmb_Onion = (T_cmb_Onion / T0) - 1.0
t_cmb_Onion_anios = calcular_tiempo_evento(T_cmb_Onion, H_func_Onion_base) / (3600 * 24 * 365.25)
#t_z10_anios = calcular_tiempo_evento(10, H_func_Onion_base) / (3600 * 24 * 365.25)

chi_Onion = calcular_distancia_comovil(T_cmb_Onion, H_func_Onion_base, gss0=gss0_Onion_base)
Omega_k0_Onion = Oml0_Onion_base - 1.0
DM_Onion = calcular_DM_Onion(chi_Onion, Oml0_Onion_base)
DA_Onion = DM_Onion / (1.0 + z_cmb_Onion)
theta_rad_Onion = (rs_LCDM / (1.0 + z_cmb_Onion)) / DA_Onion

print("\n" + "="*65)
print(" RESULTADOS ONION BASE ")
print("="*65)
print(f"Pico Visibilidad (CMB): z = {z_cmb_Onion:.0f} | t = {t_cmb_Onion_anios:,.0f} años")
#print(f"Años en z=10: {t_z10_anios:,.0f}")
print(f"Dist. comóvil D_M: {DM_Onion:,.2f} Mpc")
print(f"Dist. angular D_A: {DA_Onion:,.2f} Mpc")
print(f"Ángulo theta: {theta_rad_Onion:.5f} rad")

#%% GRÁFICAS DE LAS RELACIONES TIEMPO-TEMPERATURA-FACTOR DE ESCALA
"""
T_max = 1.22e21
T_full = np.geomspace(T_max, T0, 2000)
tLCDM_full, tOnion_full = tiempo_integracion(T_full, H_func_LCDM, H_func_Onion_base)
tOnion2_full = Tiempo_Onion_Analitico(T_full)

segundos_por_ano = 3600 * 24 * 365.25
tLCDM_full_yr = tLCDM_full / segundos_por_ano
tOnion_full_yr = tOnion_full / segundos_por_ano
tOnion2_full_yr = tOnion2_full / segundos_por_ano
a_full = T0 / T_full

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1
ax1.plot(T_full, tOnion2_full_yr, label='Onion (Analytic)', color='#FFFF99', linestyle='-', linewidth=6)
ax1.plot(T_full, tOnion_full_yr, label='Onion (Integration)', color='red', linestyle='--', linewidth=1.5)
ax1.plot(T_full, tLCDM_full_yr, label=r'$\Lambda$CDM (Integration)', color='blue', linestyle='-', linewidth=1.5)
ax1.set_xscale('log')
ax1.set_yscale('log')
yticks =[1e10, 1, 1e-10, 1e-20, 1e-30, 1e-40, 1e-50]
ax1.set_yticks(yticks)
ax1.invert_xaxis() # Solo invertimos la temperatura

# TAMAÑOS AUMENTADOS AQUÍ
ax1.set_xlabel('Temperature $T$ (MeV)', fontsize=16)
ax1.set_ylabel('Time $t$ (yr)', fontsize=16)
ax1.set_title('Age-temperature relationship', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14) # Tamaño de los números
ax1.legend(loc='upper left', frameon=True, edgecolor='black', facecolor='white', fontsize=12)
ax1.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)

# Panel 2
ax2.plot(tOnion2_full_yr, a_full, label='Onion (Analytic)', color='#FFFF99', linestyle='-', linewidth=6)
ax2.plot(tOnion_full_yr, a_full, label='Onion (Integration)', color='red', linestyle='--', linewidth=1.5)
ax2.plot(tLCDM_full_yr, a_full, label=r'$\Lambda$CDM (Integration)', color='blue', linestyle='-', linewidth=1.5)
ax2.set_xscale('log')
ax2.set_yscale('log')

# TAMAÑOS AUMENTADOS AQUÍ
ax2.set_xlabel('Time $t$ (yr)', fontsize=16)
ax2.set_ylabel('Scale Factor $a$', fontsize=16)
ax2.set_title('Age-scale factor relationship', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14) # Tamaño de los números
ax1.legend(loc='lower right', frameon=True, edgecolor='black', facecolor='white', fontsize=16)
ax2.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('panel_expansion_completo.pdf', format='pdf', bbox_inches='tight')
plt.show()
"""

#%% CÁLCULO DE LOS Z_CMB PARA LCDM Y PARA ONION CON UN VALOR DADO DE OMEGA
archivos_eta_todos = glob.glob("resultados_abundancias*.csv")
archivo_eta_lcdm = max([f for f in archivos_eta_todos if 'LCDM' in f.upper()], key=os.path.getmtime)
archivo_eta_onion = max([f for f in archivos_eta_todos if 'LCDM' not in f.upper()], key=os.path.getmtime)
print(f"Cargando LCDM desde: {archivo_eta_lcdm}")
print(f"Cargando Onion desde: {archivo_eta_onion}")
df_eta_lcdm = pd.read_csv(archivo_eta_lcdm).sort_values(by='eta').reset_index(drop=True)
df_eta_onion = pd.read_csv(archivo_eta_onion).sort_values(by='eta').reset_index(drop=True)

eta_lcdm_plot_list =[]
z_cmb_lcdm_plot_list =[]
for index, row in df_eta_lcdm.iterrows():
    eta_test = row['eta']
    x_p_test = row['p'] 
    yp_test = 4.0 * row['4He']
    yd_test = 2.0 * row['d']
    y3he_test = 3.0 * row['3He']

    interp_Xe_test = resolver_recombinacion_peebles(eta_test, x_p_test, yp_test, yd_test, y3he_test, H_func_LCDM)
    _, T_cmb_test, _ = calcular_dinamica_CMB(T_grid, lnT_grid, interp_Xe_test, eta_test, x_p_test, yp_test, yd_test, y3he_test, H_func_LCDM)
    z_test = (T_cmb_test / T0) - 1.0

    eta_lcdm_plot_list.append(eta_test)
    z_cmb_lcdm_plot_list.append(z_test)
    
Omega_search = 0.9706
df_eta_onion_2 = df_eta_onion[df_eta_onion['Omega_L0'] == Omega_search].reset_index(drop=True)

eta_onion_plot_list =[]
z_cmb_onion_plot_list =[]

for index, row in df_eta_onion_2.iterrows():
    eta_test = row['eta']
    x_p_test = row['p'] 
    yp_test = 4.0 * row['4He']
    yd_test = 2.0 * row['d']
    y3he_test = 3.0 * row['3He']
    T_nu_test = row['T_nu']
    
    H_func_test = lambda lnT: H_Onion(lnT, Omega_L0=Omega_search)

    interp_Xe_test = resolver_recombinacion_peebles(eta_test, x_p_test, yp_test, yd_test, y3he_test, H_func_test)
    _, T_cmb_test, _ = calcular_dinamica_CMB(T_grid, lnT_grid, interp_Xe_test, eta_test, x_p_test, yp_test, yd_test, y3he_test, H_func_test)
    z_test = (T_cmb_test / T0) - 1.0

    eta_onion_plot_list.append(eta_test)
    z_cmb_onion_plot_list.append(z_test)

#%% GRÁFICA DE Z_CMB PARA LCDM Y ONION
"""
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(eta_onion_plot_list, z_cmb_onion_plot_list, color='darkorange', lw=3, label=fr'$z_{{CMB}}$ (Onion, $\Omega_{{\Lambda 0}} = {Omega_search:.2f}$)')
plt.plot(eta_lcdm_plot_list, z_cmb_lcdm_plot_list, color='royalblue', lw=3, label=r'$z_{CMB}$ ($\Lambda$CDM)')

# Ajuste de Hu & Sugiyama
eta_fit = np.logspace(-10, -9, 1000)
omega_b = (eta_fit * 1e10) / 274.0
omega_m = 0.143
g1 = (0.0783 * omega_b**-0.238) / (1.0 + 39.5 * omega_b**0.763)
g2 = 0.560 / (1.0 + 21.1 * omega_b**1.81)
z_star = 1048 * (1.0 + 0.00124 * omega_b**-0.738) * (1.0 + g1 * omega_m**g2)
plt.plot(eta_fit, z_star, color='mediumseagreen', linestyle='-.', lw=2.5, label=r'$z_{CMB}$ fit (Hu & Sugiyama 1996)')
plt.axhline(z_cmb_LCDM, color='gray', linestyle='--', lw=2, alpha=0.7, label=fr'$z_{{CMB}}$ LCDM ($\eta = {eta_LCDM}$) $\approx$ {z_cmb_LCDM:.0f}')
plt.xscale('log')
plt.grid(True, which="both", alpha=0.4, linestyle='--')

# TAMAÑOS AUMENTADOS AQUÍ
plt.xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
plt.xlim(1e-10,1e-6)
plt.ylabel(r'$z_{CMB}$', fontsize=25)
plt.legend(loc='best', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20) # Tamaño de los números de los ejes

plt.tight_layout()
plt.savefig(f"grafica_zcmb_vs_eta_{os.path.splitext(os.path.basename(archivo_eta_onion))[0]}.pdf", format='pdf', bbox_inches='tight')
"""

#%% GRÁFICAS DE TAU Y FRACCIÓN DE IONIZACIÓN (X_e) LADO A LADO
"""
Xe_LCDM_plot = interp_Xe_LCDM(T_grid)
Xe_Onion_plot = interp_Xe_Onion(T_grid)
z_grid = (T_grid / T0) - 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=200)

# =============================================================================
# PANEL IZQUIERDO: Profundidad Óptica (Tau)
# =============================================================================
ax1.plot(z_grid, tau_LCDM, label=r'$\Lambda$CDM', color='royalblue', lw=2.5)
ax1.plot(z_grid, tau_Onion, label=fr'Onion ($\Omega_{{\Lambda 0}} = {Oml0_Onion_base:.2f}$, $\eta = {eta_Onion:.2e}$)', color='tomato', lw=2.5)

ax1.axhline(1.0, color='black', linestyle='--', lw=1.5, alpha=0.8, label=r'$\tau = 1$ (CMB)')
ax1.axvline(z_cmb_LCDM, color='royalblue', linestyle=':', lw=1.5, alpha=0.6)
ax1.axvline(z_cmb_Onion, color='tomato', linestyle=':', lw=1.5, alpha=0.6)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, which="both", alpha=0.4, linestyle='--')

ax1.set_xlabel(r'Redshift $z$', fontsize=30)
ax1.set_ylabel(r'Optical depth $\tau$', fontsize=30)
ax1.legend(loc='lower left', fontsize=23)
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.invert_xaxis()

# =============================================================================
# PANEL DERECHO: Fracción de Ionización (X_e)
# =============================================================================
ax2.plot(z_grid, Xe_LCDM_plot, label=r'$X_e$ ($\Lambda$CDM)', color='royalblue', lw=2.5)
ax2.plot(z_grid, Xe_Onion_plot, label=fr'$X_e$ (Onion, $\Omega_{{\Lambda 0}} = {Oml0_Onion_base:.2f}$, $\eta = {eta_Onion:.2e}$)', color='tomato', lw=2.5)

ax2.axvline(z_cmb_LCDM, color='royalblue', linestyle=':', lw=1.5, alpha=0.6)
ax2.axvline(z_cmb_Onion, color='tomato', linestyle=':', lw=1.5, alpha=0.6)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which="both", alpha=0.4, linestyle='--')

ax2.set_xlabel(r'Redshift $z$', fontsize=30)
ax2.set_ylabel(r'Ionization fraction $X_e$', fontsize=30)
#ax2.legend(loc='upper right', fontsize=17) 
ax2.tick_params(axis='both', which='major', labelsize=25)
ax2.invert_xaxis()

# Ajuste final y guardado
plt.tight_layout()
fig.subplots_adjust(wspace=0.2) 
plt.savefig("grafica_tau_y_Xe_vs_z.pdf", format='pdf', bbox_inches='tight')
plt.show()
"""

#%% GRÁFICA DE DISTANCIAS COMÓVIL Y ANGULAR VS REDSHIFT (MÚLTIPLES OMEGAS)

omega_values_to_plot =[0.0, 0.52, 0.85, Oml0_Onion_base, 0.9999] 
omega_values_to_plot = sorted(list(set(omega_values_to_plot))) # Ordena y quita duplicados si los hubiera

z_arr = np.geomspace(0.01, 1500, 1000)
z_dots = np.geomspace(0.01, 1500, 25)

T_arr = T0 * (1.0 + z_arr)

chi_arr_LCDM = np.array([calcular_distancia_comovil(T_val, H_func_LCDM, gss0=3.938) for T_val in T_arr])
DM_arr_LCDM = np.array([calcular_DM_LCDM(chi) for chi in chi_arr_LCDM])
DA_arr_LCDM = DM_arr_LCDM / (1.0 + z_arr)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, dpi=300)

ax1.plot(z_arr, DM_arr_LCDM/rs_LCDM, label=r'$\Lambda$CDM', color='blue', lw=3, zorder=10)
ax2.plot(z_arr, DA_arr_LCDM/rs_LCDM, label=r'$\Lambda$CDM', color='blue', lw=3, zorder=10)
ax1.axvline(z_cmb_LCDM, color='blue', linestyle=':', lw=1.5, alpha=0.5)
ax1.plot(z_cmb_LCDM, DM_LCDM/rs_LCDM, 'bo', markersize=8, markeredgecolor='black', zorder=15)
ax2.axvline(z_cmb_LCDM, color='blue', linestyle=':', lw=1.5, alpha=0.5)
ax2.plot(z_cmb_LCDM, DA_LCDM/rs_LCDM, 'bo', markersize=8, markeredgecolor='black', zorder=15)

colores_onion = cm.OrRd(np.linspace(0.3, 1.0, len(omega_values_to_plot)))

for idx, om_val in enumerate(omega_values_to_plot):
    H_func_temp = lambda lnT, ov=om_val: H_Onion(lnT, Omega_L0=ov)
    
    # 1. Integración Numérica (líneas continuas/discontinuas)
    chi_arr_temp = np.array([calcular_distancia_comovil(T_val, H_func_temp, gss0=gss0_Onion_base) for T_val in T_arr])
    DM_arr_temp = np.array([calcular_DM_Onion(chi, om_val) for chi in chi_arr_temp])
    DA_arr_temp = DM_arr_temp / (1.0 + z_arr)
    
    # 2. Solución Analítica (puntos)
    DM_analitico = calcular_DM_Onion_Analitico(z_dots, om_val)
    DA_analitico = DM_analitico / (1.0 + z_dots)
    
    if np.isclose(om_val, Oml0_Onion_base):
        grosor = 3.0
        estilo = '-'
        alfa = 1.0
        color_linea = 'red' 
        
        ax1.axvline(z_cmb_Onion, color=color_linea, linestyle=':', lw=1.5, alpha=0.5)
        ax1.plot(z_cmb_Onion, DM_Onion/rs_LCDM, 'ro', markersize=8, markeredgecolor='black', zorder=15)
        ax2.axvline(z_cmb_Onion, color=color_linea, linestyle=':', lw=1.5, alpha=0.5)
        ax2.plot(z_cmb_Onion, DA_Onion/rs_LCDM, 'ro', markersize=8, markeredgecolor='black', zorder=15)
        
        # Etiqueta para la versión analítica (solo la añadimos una vez para no saturar la leyenda)
        label_analitico = 'Onion analytical'
    else:
        grosor = 1.5
        estilo = '--'
        alfa = 0.7
        color_linea = colores_onion[idx]
        label_analitico = None

    # Trazado de las líneas numéricas
    ax1.plot(z_arr, DM_arr_temp/rs_LCDM, label=fr'Onion ($\Omega_{{\Lambda 0}}={om_val:.2f}$)', 
             color=color_linea, lw=grosor, linestyle=estilo, alpha=alfa)
    ax2.plot(z_arr, DA_arr_temp/rs_LCDM, color=color_linea, lw=grosor, linestyle=estilo, alpha=alfa)
    
    # Trazado de los puntos analíticos
    ax1.plot(z_dots, DM_analitico/rs_LCDM, marker='o', markersize=8, color=color_linea, 
             linestyle='none', alpha=alfa, label=label_analitico)
    ax2.plot(z_dots, DA_analitico/rs_LCDM, marker='o', markersize=8, color=color_linea, 
             linestyle='none', alpha=alfa)

ax1.set_ylabel(r'Comoving distance $D_M/r_s$', fontsize=25)
ax1.grid(True, which="both", alpha=0.4, linestyle='--')
ax1.set_ylim(0.0,300)
# Ajustamos ligeramente el tamaño de la fuente de la leyenda para que quepa bien el nuevo label
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
    
plt.savefig("distancias_multiples.pdf", format='pdf', bbox_inches='tight')
plt.show()


#%% CÁLCULO DE LA RECOMBINACIÓN PARA EL ARCHIVO DEL GRID 2D
total_pts = len(df_eta_onion)
eta_list = []
omega_list = []
Yp_list =[]
D_H_list = []  
theta_list =[]
start_time = time.time()

for i, row in df_eta_onion.iterrows():
    eta_val = row['eta']
    omega_val = row['Omega_L0']
    T_nu_val = row['T_nu']
    Yp_val = 4.0 * row['4He']
    D_H_val = row['d'] / row['p']
    x_p_val = 1.0 * row['p']
    yd_val = 2.0 * row['d']
    y3he_val = 3.0 * row['3He']
    _, gss0_test = calc_g_star_limits(T_nu_val)
    H_func_test = lambda lnT, ov=omega_val: H_Onion(lnT, Omega_L0=ov)

    theta_test = np.nan
    
    interp_Xe_test = resolver_recombinacion_peebles(eta_val, x_p_val, Yp_val, yd_val, y3he_val, H_func_test)
    _, T_cmb_test, _ = calcular_dinamica_CMB(T_grid, lnT_grid, interp_Xe_test, eta_val, x_p_val, Yp_val, yd_val, y3he_val, H_func_test)
    z_test = (T_cmb_test / T0) - 1.0
    
    # CORREGIDO: Aplicamos la geometría esférica a la distancia
    chi_test = calcular_distancia_comovil(T_cmb_test, H_func_test, gss0=gss0_test)
    Omega_k0_test = omega_val - 1.0
    DM_test = calcular_DM_Onion(chi_test, omega_val)
    
    theta_test = (rs_LCDM / (1.0 + z_test)) / (DM_test / (1.0 + z_test))

    eta_list.append(eta_val)
    omega_list.append(omega_val)
    Yp_list.append(Yp_val)
    D_H_list.append(D_H_val) 
    theta_list.append(theta_test)

    if (i + 1) % 10 == 0 or (i + 1) == total_pts:
        elap = time.time() - start_time
        rem = (elap / (i + 1)) * (total_pts - (i + 1))
        print(f" Progreso: {i+1}/{total_pts} | Tiempo est. restante: {rem/60:.1f} min", end='\r')

df_2d_fisico = pd.DataFrame({
    'eta': eta_list,
    'Omega_L0': omega_list,
    'Yp': Yp_list,
    'D_H': D_H_list,
    'theta': theta_list
})

#%% GUARDAR DICHOS DATOS EN UN .CSV
nombre_base = os.path.splitext(os.path.basename(archivo_eta_onion))[0]
archivo_guardado = f"datos_fisicos_2D_{nombre_base}.csv"
df_2d_fisico.to_csv(archivo_guardado, index=False)

#%% CARGAR DATOS DE UN .CSV
archivos_2d_fisicos = glob.glob("datos_fisicos_2D_*.csv")
archivo_carga = max(archivos_2d_fisicos, key=os.path.getmtime)
df_loaded = pd.read_csv(archivo_carga)
eta_list = df_loaded['eta'].values
omega_list = df_loaded['Omega_L0'].values
Yp_list = df_loaded['Yp'].values
D_H_list = df_loaded['D_H'].values
theta_list = df_loaded['theta'].values

nombre_plot_base = os.path.splitext(os.path.basename(archivo_carga))[0].replace("datos_fisicos_2D_", "")

#%% PHYSICAL VALUES CONTOUR PLOTS (WITH 1, 2 & 3-SIGMA LIKELIHOOD REGIONS)
theta_obs_local = 0.010411
sigma_theta_local = 0.0000046
Yp_obs_local = 0.245
sigma_Yp_local = 0.003
D_obs_local = 2.547e-5
sigma_D_local = 0.025e-5

unique_omegas = np.sort(np.unique(omega_list))

eta_log = np.log10(eta_list)
eta_dense_log = np.linspace(eta_log.min(), eta_log.max(), 1000)

Yp_dense_2d = np.zeros((len(unique_omegas), len(eta_dense_log)))
D_H_dense_2d = np.zeros((len(unique_omegas), len(eta_dense_log))) 
theta_dense_2d = np.zeros((len(unique_omegas), len(eta_dense_log)))

# Interpolation
for idx_om, om_val in enumerate(unique_omegas):
    mask = (omega_list == om_val)
    etas_om_log = eta_log[mask]
    Yp_om = Yp_list[mask]
    D_H_om = D_H_list[mask]
    theta_om = theta_list[mask]
    
    sort_idx = np.argsort(etas_om_log)
    etas_om_log = etas_om_log[sort_idx]
    Yp_om = Yp_om[sort_idx]
    D_H_om = D_H_om[sort_idx]
    theta_om = theta_om[sort_idx]
    
    f_yp = interp1d(etas_om_log, Yp_om, kind='cubic', bounds_error=False, fill_value="extrapolate")
    Yp_dense_2d[idx_om, :] = f_yp(eta_dense_log)
    
    f_D = interp1d(etas_om_log, D_H_om, kind='cubic', bounds_error=False, fill_value="extrapolate")
    D_H_dense_2d[idx_om, :] = f_D(eta_dense_log)
    
    valid_mask = ~np.isnan(theta_om)
    if np.sum(valid_mask) > 3:
        f_th = interp1d(etas_om_log[valid_mask], theta_om[valid_mask], kind='cubic', bounds_error=False, fill_value="extrapolate")
        theta_dense_2d[idx_om, :] = f_th(eta_dense_log)
    else:
        theta_dense_2d[idx_om, :] = np.nan

theta_dense_2d_100 = theta_dense_2d * 100.0
theta_obs_local_100 = 1.04105
sigma_theta_local_100 = 0.00046

Lik_Yp_2d = np.exp(-0.5 * ((Yp_dense_2d - Yp_obs_local) / sigma_Yp_local)**2)
Lik_D_2d = np.exp(-0.5 * ((D_H_dense_2d - D_obs_local) / sigma_D_local)**2)
Lik_theta_2d = np.zeros_like(theta_dense_2d_100)
valid_th_dense = ~np.isnan(theta_dense_2d_100)
Lik_theta_2d[valid_th_dense] = np.exp(-0.5 * ((theta_dense_2d_100[valid_th_dense] - theta_obs_local_100) / sigma_theta_local_100)**2)

Eta_log_mesh, Om_mesh = np.meshgrid(eta_dense_log, unique_omegas)
Eta_mesh_real = 10**Eta_log_mesh

def compute_2d_contours(likelihood_grid):
    L_flat = likelihood_grid[~np.isnan(likelihood_grid)]
    if np.sum(L_flat) == 0:
        return [1.0, 1.0, 1.0] # Si todo es cero, devuelve un límite inalcanzable
        
    L_sorted = np.sort(L_flat)[::-1]
    L_pdf = L_sorted / np.sum(L_sorted)
    cum_sum = np.cumsum(L_pdf)
    
    try: lvl_1sig = L_sorted[np.searchsorted(cum_sum, 0.6827)]
    except IndexError: lvl_1sig = L_sorted[-1]
    try: lvl_2sig = L_sorted[np.searchsorted(cum_sum, 0.9545)]
    except IndexError: lvl_2sig = L_sorted[-1]
    try: lvl_3sig = L_sorted[np.searchsorted(cum_sum, 0.9973)]
    except IndexError: lvl_3sig = L_sorted[-1]
    
    return [lvl_3sig, lvl_2sig, lvl_1sig]

contour_levels_Yp = compute_2d_contours(Lik_Yp_2d)
contour_levels_D = compute_2d_contours(Lik_D_2d)
contour_levels_theta = compute_2d_contours(Lik_theta_2d)

# Hacemos las líneas de la leyenda un poco más finas para que coincidan con el nuevo estilo
line_1sig = mlines.Line2D([],[], color='black', linestyle='-', linewidth=1.5, label=r'$1\sigma$')
line_2sig = mlines.Line2D([],[], color='black', linestyle='--', linewidth=1.2, label=r'$2\sigma$')
line_3sig = mlines.Line2D([],[], color='black', linestyle=':', linewidth=1.5, label=r'$3\sigma$')

# =============================================================================
# GRÁFICA 1: HELIO-4 (Yp)
# =============================================================================
fig_Yp, ax_Yp = plt.subplots(1, 1, figsize=(10, 5), dpi=200)

cf_Yp = ax_Yp.contourf(Eta_mesh_real, Om_mesh, Yp_dense_2d, levels=100, cmap='plasma')
if contour_levels_Yp[0] < 1.0:
    ax_Yp.contour(Eta_mesh_real, Om_mesh, Lik_Yp_2d, levels=contour_levels_Yp, 
                  colors=['black', 'black', 'black'], linestyles=[':', '--', '-'], 
                  linewidths=[0.8, 1.2, 1.5], alpha=0.9)

cb_Yp = fig_Yp.colorbar(cf_Yp, ax=ax_Yp, pad=0.02)
cb_Yp.set_label(r'$Y_p$', fontsize=24)
cb_Yp.ax.tick_params(labelsize=16)

ax_Yp.set_xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
ax_Yp.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
ax_Yp.grid(True, which="both", alpha=0.3, linestyle='--')
ax_Yp.legend(handles=[line_1sig, line_2sig, line_3sig], loc='upper right', fontsize=20)
ax_Yp.set_xscale('log')
#ax_Yp.set_xlim(2e-9,5e-9)

ax_Yp.tick_params(axis='both', which='major', labelsize=16)
ax_Yp.tick_params(axis='x', which='minor', labelsize=16)

fig_Yp.tight_layout()
fig_Yp.savefig("contour_Yp.pdf", format='pdf', bbox_inches='tight')


# =============================================================================
# GRÁFICA 2: DEUTERIO (D/H)
# =============================================================================
fig_D, ax_D = plt.subplots(1, 1, figsize=(10, 5), dpi=200)

cf_D = ax_D.contourf(Eta_mesh_real, Om_mesh, D_H_dense_2d, levels=100, cmap='cividis')
if contour_levels_D[0] < 1.0:
    ax_D.contour(Eta_mesh_real, Om_mesh, Lik_D_2d, levels=contour_levels_D, 
                 colors=['black', 'black', 'black'], linestyles=[':', '--', '-'], 
                 linewidths=[0.8, 1.2, 1.5], alpha=0.9)
cb_D = fig_D.colorbar(cf_D, ax=ax_D, pad=0.02)
cb_D.set_label(r'$D/H$', fontsize=24)
cb_D.ax.tick_params(labelsize=16)

ax_D.set_xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
ax_D.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
ax_D.grid(True, which="both", alpha=0.3, linestyle='--')
ax_D.set_xscale('log')

ax_D.tick_params(axis='both', which='major', labelsize=16)
ax_D.tick_params(axis='x', which='minor', labelsize=16)

fig_D.tight_layout()
fig_D.savefig("contour_DH.pdf", format='pdf', bbox_inches='tight')


# =============================================================================
# GRÁFICA 3: TAMAÑO ANGULAR DEL CMB (THETA)
# =============================================================================
fig_th, ax_th = plt.subplots(1, 1, figsize=(10, 5), dpi=200)

niveles_theta = np.linspace(0.0, 2.0, 100)
cf_th = ax_th.contourf(Eta_mesh_real, Om_mesh, theta_dense_2d_100, levels=niveles_theta, cmap='viridis', extend='max')

niveles_theta_exactos =[
    theta_obs_local_100 - 3*sigma_theta_local_100,
    theta_obs_local_100 - 2*sigma_theta_local_100,
    theta_obs_local_100 - 1*sigma_theta_local_100,
    theta_obs_local_100 + 1*sigma_theta_local_100,
    theta_obs_local_100 + 2*sigma_theta_local_100,
    theta_obs_local_100 + 3*sigma_theta_local_100
]

# Trazamos los contornos directamente sobre el valor físico
ax_th.contour(Eta_mesh_real, Om_mesh, theta_dense_2d_100, levels=niveles_theta_exactos, 
              colors=['black', 'black', 'black', 'black', 'black', 'black'], 
              linestyles=[':', '--', '-', '-', '--', ':'], 
              linewidths=[1.5, 1.5, 2, 2, 1.5, 1.5], alpha=0.8)

cb_th = fig_th.colorbar(cf_th, ax=ax_th, pad=0.02)
cb_th.set_label(r'$100\theta_{CMB}$', fontsize=24)
cb_th.ax.tick_params(labelsize=16)

ax_th.set_xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
ax_th.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
ax_th.grid(True, which="both", alpha=0.3, linestyle='--')
#ax_th.set_ylim(0.998,0.9999)
#ax_th.set_xlim(2e-9,5e-9)

# --- SOLUCIÓN PARA LA LEYENDA ---
# Creamos una única línea representativa para el solapamiento
line_overlap = mlines.Line2D([],[], color='black', linestyle='-', linewidth=2.5, 
                             label=r'$1\sigma, 2\sigma, 3\sigma$')
# Y se la pasamos a la leyenda en lugar de las 3 separadas
ax_th.legend(handles=[line_overlap], loc='upper right', fontsize=18)

ax_th.set_xscale('log')
ax_th.tick_params(axis='both', which='major', labelsize=16)
ax_th.tick_params(axis='x', which='minor', labelsize=16)

fig_th.tight_layout()
fig_th.savefig("contour_theta.pdf", format='pdf', bbox_inches='tight')

#%%

# =============================================================================
# CÁLCULO DE LA INTERSECCIÓN A 2-SIGMA (Yp y Theta_CMB)
# =============================================================================
om_dense = np.linspace(unique_omegas.min(), unique_omegas.max(), 5000)

# Interpolamos las matrices 2D a lo largo del eje Y (Omega)
f_Yp_interp = interp1d(unique_omegas, Yp_dense_2d, axis=0, kind='linear')
Yp_superdense = f_Yp_interp(om_dense)

f_th_interp = interp1d(unique_omegas, theta_dense_2d_100, axis=0, kind='linear')
theta_superdense = f_th_interp(om_dense)

# Creamos las mallas de coordenadas super densas
Eta_supermesh, Om_supermesh = np.meshgrid(eta_dense_log, om_dense)
Eta_supermesh_real = 10**Eta_supermesh

# 1. Máscara 2-sigma para Yp (Condición matemática exacta)
mask_Yp_2sig = np.abs(Yp_superdense - Yp_obs_local) <= 2 * sigma_Yp_local

# 2. Máscara 2-sigma para Theta (Condición matemática exacta)
mask_theta_2sig = np.abs(theta_superdense - theta_obs_local_100) <= 2 * sigma_theta_local_100

# 3. Intersección de ambas regiones
intersection_mask = mask_Yp_2sig & mask_theta_2sig

print("\n" + "="*65)
print(" INTERSECCIÓN A 2-SIGMA (Yp y Theta_CMB) ")
print("="*65)

if np.any(intersection_mask):
    # 1. "Inflamos" los píxeles de intersección usando una caja de 20x20. 
    # Así, los píxeles sueltos que pertenecen al mismo cruce se fusionarán en un solo bloque.
    mask_dilatada = ndimage.binary_dilation(intersection_mask, structure=np.ones((20, 20)))
    
    # 2. Etiquetamos los bloques fusionados
    labeled_mask, num_features = ndimage.label(mask_dilatada)
    
    print(f"Se han encontrado {num_features} región/es de intersección principal(es):")
    
    for i in range(1, num_features + 1):
        # Recuperamos SOLO los píxeles originales (sin inflar) que caen dentro de este bloque
        region_mask = (labeled_mask == i) & intersection_mask
        
        # Si por algún motivo la máscara está vacía, saltamos
        if not np.any(region_mask):
            continue
            
        eta_region = Eta_supermesh_real[region_mask]
        om_region = Om_supermesh[region_mask]
        
        eta_min, eta_max = eta_region.min(), eta_region.max()
        om_min, om_max = om_region.min(), om_region.max()
        
        # Calculamos el valor central y el error (mitad del ancho del intervalo)
        eta_center = (eta_max + eta_min) / 2.0
        eta_err = (eta_max - eta_min) / 2.0
        
        om_center = (om_max + om_min) / 2.0
        om_err = (om_max - om_min) / 2.0
        
        # Formateo elegante para eta (agrupando la notación científica)
        exp_eta = int(np.floor(np.log10(eta_center)))
        eta_c_norm = eta_center / 10**exp_eta
        eta_e_norm = eta_err / 10**exp_eta
        
        print(f"\n--- Solución / Región {i} ---")
        print(f"eta      = ({eta_c_norm:.3f} ± {eta_e_norm:.3f}) x 10^{exp_eta}")
        print(f"Omega_L0 = {om_center:.7f} ± {om_err:.7f}")
else:
    print("No hay intersección a 2-sigma entre Yp y Theta_CMB en este grid.")
    
#%%
    # =============================================================================
# GRÁFICA COMBINADA: HELIO-4 (Yp) Y TAMAÑO ANGULAR DEL CMB (THETA)
# =============================================================================
# Creamos 2 filas, 1 columna, compartiendo el eje X. Aumentamos el alto a 9.
fig, (ax_Yp, ax_th) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, dpi=200)

# -----------------------------------------------------------------------------
# PANEL SUPERIOR: HELIO-4 (Yp)
# -----------------------------------------------------------------------------
cf_Yp = ax_Yp.contourf(Eta_mesh_real, Om_mesh, Yp_dense_2d, levels=100, cmap='plasma')
if contour_levels_Yp[0] < 1.0:
    ax_Yp.contour(Eta_mesh_real, Om_mesh, Lik_Yp_2d, levels=contour_levels_Yp, 
                  colors=['black', 'black', 'black'], linestyles=[':', '--', '-'], 
                  linewidths=[0.8, 1.2, 1.5], alpha=1.0)

cb_Yp = fig.colorbar(cf_Yp, ax=ax_Yp, pad=0.02)
cb_Yp.set_label(r'$Y_p$', fontsize=24)
cb_Yp.ax.tick_params(labelsize=16)

ax_Yp.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
ax_Yp.grid(True, which="both", alpha=0.3, linestyle='--')
ax_Yp.legend(handles=[line_1sig, line_2sig, line_3sig], loc='upper right', fontsize=20)

ax_Yp.tick_params(axis='both', which='major', labelsize=16)
# No le ponemos xlabel a este panel porque lo comparte con el de abajo

# -----------------------------------------------------------------------------
# PANEL INFERIOR: TAMAÑO ANGULAR DEL CMB (THETA)
# -----------------------------------------------------------------------------
niveles_theta = np.linspace(0.0, 2.0, 100)
cf_th = ax_th.contourf(Eta_mesh_real, Om_mesh, theta_dense_2d_100, levels=niveles_theta, cmap='viridis', extend='max')

niveles_theta_exactos =[
    theta_obs_local_100 - 3*sigma_theta_local_100,
    theta_obs_local_100 - 2*sigma_theta_local_100,
    theta_obs_local_100 - 1*sigma_theta_local_100,
    theta_obs_local_100 + 1*sigma_theta_local_100,
    theta_obs_local_100 + 2*sigma_theta_local_100,
    theta_obs_local_100 + 3*sigma_theta_local_100
]

ax_th.contour(Eta_mesh_real, Om_mesh, theta_dense_2d_100, levels=niveles_theta_exactos, 
              colors=['black', 'black', 'black', 'black', 'black', 'black'], 
              linestyles=[':', '--', '-', '-', '--', ':'], 
              linewidths=[1.5, 1.5, 2, 2, 1.5, 1.5], alpha=0.8)

cb_th = fig.colorbar(cf_th, ax=ax_th, pad=0.02)
cb_th.set_label(r'$100\theta_{CMB}$', fontsize=24)
cb_th.ax.tick_params(labelsize=16)

ax_th.set_xlabel(r'Baryon-to-photon ratio $\eta$', fontsize=20)
ax_th.set_ylabel(r'$\Omega_{\Lambda 0}$', fontsize=20)
ax_th.grid(True, which="both", alpha=0.3, linestyle='--')

# Leyenda unificada para theta
line_overlap = mlines.Line2D([],[], color='black', linestyle='-', linewidth=2.5, 
                             label=r'$1\sigma, 2\sigma, 3\sigma$')
ax_th.legend(handles=[line_overlap], loc='upper right', fontsize=18)

# Configuramos el eje X (al ser sharex=True, esto afecta a ambas gráficas)
ax_th.set_xscale('log')
#ax_th.set_xlim(2e-9, 5e-9)
ax_th.tick_params(axis='both', which='major', labelsize=16)
ax_th.tick_params(axis='x', which='minor', labelsize=16)

# -----------------------------------------------------------------------------
# AJUSTES FINALES Y GUARDADO
# -----------------------------------------------------------------------------
fig.tight_layout()
# Reducimos un poco el espacio vertical entre ambas gráficas
fig.subplots_adjust(hspace=0.08)

pdf_filename = "contour_Yp_and_theta_combined.pdf"
fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
print(f"Gráfica combinada guardada como: {pdf_filename}")
