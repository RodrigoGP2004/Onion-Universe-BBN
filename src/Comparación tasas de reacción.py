import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DATOS EXPERIMENTALES (TABLAS COMPLETAS CON BARRAS DE ERROR)
# =====================================================================
T9_luna = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
R_luna = np.array([1.37e-11, 2.57e-5, 1.53e-3, 9.08e-1, 5.74, 1.29e2, 3.63e2, 6.32e2, 9.20e2, 1.52e3, 2.11e3, 2.67e3, 3.16e3, 3.56e3, 3.85e3, 4.01e3, 4.02e3])
R_luna_low = np.array([1.35e-11, 2.53e-5, 1.51e-3, 8.94e-1, 5.65, 1.26e2, 3.52e2, 6.09e2, 8.79e2, 1.43e3, 1.95e3, 2.40e3, 2.76e3, 3.00e3, 3.09e3, 3.02e3, 2.75e3])
R_luna_high = np.array([1.39e-11, 2.62e-5, 1.56e-3, 9.22e-1, 5.84, 1.32e2, 3.74e2, 6.56e2, 9.62e2, 1.61e3, 2.28e3, 2.93e3, 3.55e3, 4.12e3, 4.61e3, 5.01e3, 5.30e3])
err_luna = np.vstack((R_luna - R_luna_low, R_luna_high - R_luna))

T9_gomez = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.018, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.180, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.700, 0.800, 0.900, 1.000, 1.250, 1.500, 1.750, 2.000, 2.500, 3.000, 3.500, 4.000, 5.000, 6.000, 7.000, 8.000, 9.000, 10.000])
R_ddn = np.array([1.322e-08, 5.489e-05, 3.025e-03, 3.737e-02, 2.214e-01, 8.556e-01, 2.508e+00, 6.074e+00, 1.280e+01, 2.427e+01, 4.242e+01, 6.945e+01, 1.078e+02, 1.602e+02, 2.293e+02, 3.183e+02, 5.674e+02, 9.321e+02, 2.507e+03, 5.307e+03, 1.570e+04, 3.373e+04, 6.020e+04, 9.539e+04, 1.392e+05, 1.914e+05, 2.516e+05, 3.194e+05, 3.943e+05, 4.759e+05, 5.638e+05, 6.575e+05, 7.568e+05, 9.702e+05, 1.201e+06, 1.843e+06, 2.555e+06, 3.318e+06, 4.118e+06, 4.944e+06, 5.788e+06, 7.510e+06, 9.251e+06, 1.099e+07, 1.271e+07, 1.440e+07, 1.850e+07, 2.236e+07, 2.599e+07, 2.938e+07, 3.546e+07, 4.093e+07, 4.585e+07, 5.031e+07, 5.816e+07, 6.488e+07, 7.072e+07, 7.583e+07, 8.037e+07, 8.437e+07])
fu_ddn = np.array([1.011]*50 + [1.012, 1.014, 1.014, 1.015, 1.016, 1.017, 1.018, 1.018, 1.018, 1.018])
err_ddn = np.vstack((R_ddn - R_ddn/fu_ddn, R_ddn*fu_ddn - R_ddn))

R_ddp = np.array([1.364e-08, 5.653e-05, 3.110e-03, 3.835e-02, 2.269e-01, 8.755e-01, 2.563e+00, 6.198e+00, 1.304e+01, 2.471e+01, 4.314e+01, 7.055e+01, 1.094e+02, 1.624e+02, 2.322e+02, 3.220e+02, 5.729e+02, 9.395e+02, 2.516e+03, 5.305e+03, 1.558e+04, 3.325e+04, 5.900e+04, 9.298e+04, 1.350e+05, 1.847e+05, 2.418e+05, 3.056e+05, 3.758e+05, 4.518e+05, 5.334e+05, 6.199e+05, 7.111e+05, 9.061e+05, 1.116e+06, 1.691e+06, 2.321e+06, 2.988e+06, 3.681e+06, 4.391e+06, 5.113e+06, 6.573e+06, 8.036e+06, 9.489e+06, 1.092e+07, 1.233e+07, 1.572e+07, 1.893e+07, 2.194e+07, 2.477e+07, 2.976e+07, 3.440e+07, 3.863e+07, 4.251e+07, 4.946e+07, 5.552e+07, 6.077e+07, 6.529e+07, 6.912e+07, 7.228e+07])
fu_ddp = np.array([1.011]*50 + [1.013, 1.014, 1.014, 1.015, 1.016, 1.017, 1.018, 1.018, 1.018, 1.019])
err_ddp = np.vstack((R_ddp - R_ddp/fu_ddp, R_ddp*fu_ddp - R_ddp))

# =====================================================================
# 2. FUNCIONES REACLIB Y COEFICIENTES JINA ORIGINALES
# =====================================================================
def reaclib_7(T9, a0, a1, a2, a3, a4, a5, a6):
    T9_safe = np.maximum(T9, 1e-9)
    t13 = np.cbrt(T9_safe)
    t53 = T9_safe * t13**2
    lnt9 = np.log(T9_safe)
    return a0 + a1/T9_safe + a2/t13 + a3*t13 + a4*T9_safe + a5*t53 + a6*lnt9

def get_rates(T9, coefs_list):
    rate = np.zeros_like(T9)
    for c in coefs_list:
        rate += np.exp(np.clip(reaclib_7(T9, *c), -700, 100))
    return rate

# JINA Originales
jina_dp = [[7.52898, 0, -3.7208, 0.871782, 0, 0, -0.666667],
           [8.93525, 0, -3.7208, 0.198654, 0, 0, 0.333333]]
jina_dp_inv = [[31.032, -63.7435, -3.7208, 0.871782, 0.0, 0.0, 0.833333],
               [32.4383, -63.7435, -3.7208, 0.198654, 0.0, 0.0, 1.83333]]

jina_ddn = [[19.0876, -1.9002e-4, -4.2292, 1.6932, -0.0855529, -1.35709e-25, -0.734513]]
jina_ddn_inv = [[19.6369, -37.9358, -4.2292, 1.6932, -0.0855529, -1.35709e-25, -0.734513]]

jina_ddp = [[18.8052, 4.36209e-5, -4.32296, 1.91572, -0.081562, -3.28804e-22, -0.879518]]
jina_ddp_inv = [[19.3545, -46.799, -4.32296, 1.91572, -0.081562, -3.28804e-22, -0.879518]]

# =====================================================================
# 3. AJUSTE EXACTO MEDIANTE ÁLGEBRA LINEAL (7 PARÁMETROS LIBRES)
# =====================================================================
# Al ser la fórmula de Reaclib lineal respecto a sus coeficientes,
# usamos Mínimos Cuadrados Lineales (SVD) para encontrar el ajuste 
# perfecto sin iteraciones, sin errores de covarianza y sin atascos.

def fit_reaclib_7_free(T9, R):
    t13 = np.cbrt(T9)
    t53 = T9 * t13**2
    lnt9 = np.log(T9)
    
    # Matriz de diseño con los 7 términos de la ecuación de Reaclib
    X = np.column_stack([np.ones_like(T9), 1/T9, 1/t13, t13, T9, t53, lnt9])
    Y = np.log(R)
    
    # Resolvemos el sistema lineal de forma exacta
    coefs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return [coefs.tolist()]

# Calculamos los nuevos polinomios (1 componente con 7 parámetros libres)
new_luna_dir = fit_reaclib_7_free(T9_luna, R_luna)
new_ddn_dir = fit_reaclib_7_free(T9_gomez, R_ddn)
new_ddp_dir = fit_reaclib_7_free(T9_gomez, R_ddp)

# Balance Detallado para inversas (El desplazamiento termodinámico es constante)
def apply_detailed_balance(new_dir, jina_dir, jina_inv):
    shift = [jina_inv[i] - jina_dir[i] for i in range(7)]
    return [[new_dir[0][i] + shift[i] for i in range(7)]]

new_luna_inv = apply_detailed_balance(new_luna_dir, jina_dp[0], jina_dp_inv[0])
new_ddn_inv = apply_detailed_balance(new_ddn_dir, jina_ddn[0], jina_ddn_inv[0])
new_ddp_inv = apply_detailed_balance(new_ddp_dir, jina_ddp[0], jina_ddp_inv[0])

# =====================================================================
# 4. GRÁFICA (CON AJUSTES VISUALES: LÍMITE Y + TRANSPARENCIA)
# =====================================================================
T_MeV_cont = np.logspace(np.log10(1), np.log10(1e-3), 500)
T9_cont = T_MeV_cont * 11.6045

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 12})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
step = 3

# ---- PANEL IZQUIERDO: JINA (Curvas) + DATOS EXPERIMENTALES (Puntos) ----
ax1.loglog(T_MeV_cont, get_rates(T9_cont, jina_dp), color='C0', linestyle='-', alpha=0.5, label=r'JINA $d(p,\gamma)^3\mathrm{He}$')
ax1.loglog(T_MeV_cont, get_rates(T9_cont, jina_ddn), color='C1', linestyle='-', alpha=0.5, label=r'JINA $d(d,n)^3\mathrm{He}$')
ax1.loglog(T_MeV_cont, get_rates(T9_cont, jina_ddp), color='C2', linestyle='-', alpha=0.5, label=r'JINA $d(d,p)^3\mathrm{H}$')

ax1.errorbar(T9_luna/11.6045, R_luna, yerr=err_luna, fmt='o', color='C0', markersize=5, capsize=4, label=r'LUNA (2020) $d(p,\gamma)^3\mathrm{He}$')
ax1.errorbar(T9_gomez[::step]/11.6045, R_ddn[::step], yerr=err_ddn[:, ::step], fmt='s', color='C1', markersize=5, capsize=4, label=r'Gómez Iñesta $d(d,n)^3\mathrm{He}$')
ax1.errorbar(T9_gomez[::step]/11.6045, R_ddp[::step], yerr=err_ddp[:, ::step], fmt='^', color='C2', markersize=5, capsize=4, label=r'Gómez Iñesta $d(d,p)^3\mathrm{H}$')

ax1.set_xlim(1, 0.001)
ax1.set_ylim(bottom=1e-3) # <-- NUEVO LÍMITE Y EN EL PANEL IZQUIERDO
ax1.set_xlabel(r'Temperature $T$ [MeV]')
ax1.set_ylabel(r'Reaction rate $N_A \langle \sigma v \rangle$ [cm$^3$ s$^{-1}$ mol$^{-1}$]')
ax1.set_title('Updated reaction Rates vs JINA Database')
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ---- PANEL DERECHO: RATIO PUNTOS/JINA + RATIO NUEVOFIT/JINA ----
# 1. Puntos (Datos experimentales / JINA) -> Añadido alpha=0.5 para transparencia
ax2.errorbar(T9_luna/11.6045, R_luna / get_rates(T9_luna, jina_dp), 
             yerr=err_luna / get_rates(T9_luna, jina_dp), fmt='o', color='C0', markersize=5, capsize=4, alpha=0.9, label=r'Data: $d(p,\gamma)^3\mathrm{He}$')
ax2.errorbar(T9_gomez[::step]/11.6045, R_ddn[::step] / get_rates(T9_gomez[::step], jina_ddn), 
             yerr=err_ddn[:, ::step] / get_rates(T9_gomez[::step], jina_ddn), fmt='s', color='C1', markersize=5, capsize=4, alpha=0.9, label=r'Data: $d(d,n)^3\mathrm{He}$')
ax2.errorbar(T9_gomez[::step]/11.6045, R_ddp[::step] / get_rates(T9_gomez[::step], jina_ddp), 
             yerr=err_ddp[:, ::step] / get_rates(T9_gomez[::step], jina_ddp), fmt='^', color='C2', markersize=5, capsize=4, alpha=0.9, label=r'Data: $d(d,p)^3\mathrm{H}$')

# 2. Curvas continuas (Nuevos Fits polinómicos / JINA)
ax2.plot(T_MeV_cont, get_rates(T9_cont, new_luna_dir) / get_rates(T9_cont, jina_dp), color='C0', linestyle='--', linewidth=2.5, label=r'Fit: $d(p,\gamma)^3\mathrm{He}$')
ax2.plot(T_MeV_cont, get_rates(T9_cont, new_ddn_dir) / get_rates(T9_cont, jina_ddn), color='C1', linestyle='--', linewidth=2.5, label=r'Fit: $d(d,n)^3\mathrm{He}$')
ax2.plot(T_MeV_cont, get_rates(T9_cont, new_ddp_dir) / get_rates(T9_cont, jina_ddp), color='C2', linestyle='--', linewidth=2.5, label=r'Fit: $d(d,p)^3\mathrm{H}$')

ax2.set_xscale('log')
ax2.set_xlim(1, 0.001)
ax2.set_ylim(0.90, 1.25)
ax2.set_xlabel(r'Temperature $T$ [MeV]')
ax2.set_ylabel('Ratio relative to JINA Data')
ax2.set_title('Updated fits relative to JINA Fits')
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(loc='upper right', ncol=1, fontsize=14)

plt.tight_layout()
plt.savefig('comparation_rates.pdf', format='pdf', bbox_inches='tight')
print("Gráfica generada con éxito: 'comparation_rates.pdf'\n")

# =====================================================================
# 5. IMPRESIÓN DEL CÓDIGO C
# =====================================================================
def format_c(name, arr_list):
    lines = []
    for arr in arr_list:
        lines.append("{" + ", ".join([f"{x:g}" for x in arr]) + "}")
    return f"double {name}[{len(arr_list)}][7] = {{{', '.join(lines)}}};"

print("// --- CÓDIGO ACTUALIZADO PARA bbn.c ---")
print(format_c("p_dp_3He", new_luna_dir))
print("r_array[1][k] = jina_dict(T9, 1, p_dp_3He);")
print(format_c("p_photodes_3He", new_luna_inv))
print("r_array[17][k] = jina_dict(T9, 1, p_photodes_3He);\n")

print(format_c("p_dd_n3He", new_ddn_dir))
print("r_array[2][k] = jina_dict(T9, 1, p_dd_n3He);")
print(format_c("p_n3He_dd", new_ddn_inv))
print("r_array[19][k] = jina_dict(T9, 1, p_n3He_dd);\n")

print(format_c("p_dd_pt", new_ddp_dir))
print("r_array[3][k] = jina_dict(T9, 1, p_dd_pt);")
print(format_c("p_pt_dd", new_ddp_inv))
print("r_array[18][k] = jina_dict(T9, 1, p_pt_dd);")