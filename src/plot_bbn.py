import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import fsolve 
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import sys
import glob
import os

mass_pesados = {
    '9Be': 9, '12C': 12, '13C': 13, '13N': 13, '14O': 14, '14N': 14,
    '15N': 15, '15O': 15, '16O': 16, '17F': 17, '17O': 17, '18F': 18,
    '18O': 18, '18Ne': 18, '19F': 19, '20Ne': 20, '24Mg': 24, '28Si': 28,
    '32S': 32, '36Ar': 36, '40Ca': 40, '44Ti': 44
}

lw = 2.5 


archivos_csv = glob.glob("resultados_bbn*.csv")
if not archivos_csv:
    print("\nNota: No se encontró ningún archivo 'resultados_bbn*.csv'. Omitiendo gráficas de evolución BBN.")
else:
    archivo_reciente = max(archivos_csv, key=os.path.getmtime)
    print(f"Leyendo datos BBN desde: {archivo_reciente}")

    try:
        df_bbn = pd.read_csv(archivo_reciente)
        
        T_MeV = df_bbn['T_MeV'].values
        t_num = df_bbn['t'].values

        df_bbn['Pesados'] = 0
        for el, A in mass_pesados.items():
            if el in df_bbn.columns:
                df_bbn['Pesados'] += A * df_bbn[el]

        X_n  = 1 * df_bbn['n'].iloc[-1]
        X_p  = 1 * df_bbn['p'].iloc[-1]
        X_d  = 2 * df_bbn['d'].iloc[-1]
        X_t  = 3 * df_bbn['t_iso'].iloc[-1]
        X_3He= 3 * df_bbn['3He'].iloc[-1]
        X_4He= 4 * df_bbn['4He'].iloc[-1]
        X_7Li= 7 * df_bbn['7Li'].iloc[-1]
        X_7Be= 7 * df_bbn['7Be'].iloc[-1]
        X_8B = 8 * df_bbn['8B'].iloc[-1]
        X_Pesados = df_bbn['Pesados'].iloc[-1]

        print("# ABUNDANCIAS FINALES BBN (Fracciones de Masa X_i)")
        print(f"Neutrones (n)    : {X_n:.6e}")
        print(f"Protones (p)     : {X_p:.6e}")
        print(f"Deuterio (D)     : {X_d:.6e}")
        print(f"Tritio (T)       : {X_t:.6e}")
        print(f"Helio-3 (3He)    : {X_3He:.6e}")
        print(f"Helio-4 (4He)    : {X_4He:.6e}")
        print(f"Litio-7 (7Li)    : {X_7Li:.6e}")
        print(f"Berilio-7 (7Be)  : {X_7Be:.6e}")
        print(f"Boro-8 (8B)      : {X_8B:.6e}")
        print(f"Pesados (A >= 9) : {X_Pesados:.6e}")
        print("="*50 + "\n")

        print(f"2H/1H: {(df_bbn['d'].iloc[-1]/df_bbn['p'].iloc[-1]):.6e}")
        print(f"7Li/1H: {((df_bbn['7Li'].iloc[-1]+df_bbn['7Be'].iloc[-1])/df_bbn['p'].iloc[-1]):.6e}")

        plt.figure(1, figsize=(10, 6))

        plt.loglog(T_MeV, 1 * df_bbn['n'], label='n', color='black', linestyle='--', linewidth=lw)
        plt.loglog(T_MeV, 1 * df_bbn['p'], label='p', color='blue', linewidth=lw)
        plt.loglog(T_MeV, 2 * df_bbn['d'], label='D', color='green', linewidth=lw)
        plt.loglog(T_MeV, 3 * df_bbn['t_iso'], label='T', color='orange', linewidth=lw)
        plt.loglog(T_MeV, 3 * df_bbn['3He'], label='$^3$He', color='red', linewidth=lw)
        plt.loglog(T_MeV, 4 * df_bbn['4He'], label='$^4$He', color='cyan', linewidth=lw)
        plt.loglog(T_MeV, 7 * df_bbn['7Li'], label='$^7$Li', color='magenta', linewidth=lw)
        plt.loglog(T_MeV, 7 * df_bbn['7Be'], label='$^7$Be', color='brown', linewidth=lw)
        plt.loglog(T_MeV, 8 * df_bbn['8B'], label='$^8$B', color='gray', linewidth=lw)
        plt.loglog(T_MeV, df_bbn['Pesados'], label=r'A $\geq$ 9', color='purple', linestyle='-', linewidth=lw) 

        plt.xlabel('Temperature $T$ (MeV)', fontsize=20)
        plt.ylabel('Mass Fraction $X_i$', fontsize=20)
        plt.gca().invert_xaxis() 
        plt.grid(True, which="both", alpha=0.4, linestyle='--')
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=20)
        plt.xlim(1.0, np.min(T_MeV))
        plt.ylim(1e-25, 2) 
        plt.tight_layout()

        nombre_base = os.path.splitext(os.path.basename(archivo_reciente))[0]
        pdf_filename = f"grafica_{nombre_base}.pdf"
        
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        print(f"Gráfica de temperatura guardada como: {pdf_filename}")

    except Exception as e:
        print(f"Error al procesar el archivo {archivo_reciente}: {e}. Omitiendo Parte 1.")

archivos_eta = glob.glob("resultados_abundancias*.csv")

if not archivos_eta:
    print("\nNota: No se encontró ningún archivo 'resultados_abundancias*.csv'. Omitiendo gráficas de barrido de eta.")
else:
    archivo_eta_reciente = max(archivos_eta, key=os.path.getmtime)
    print(f"Leyendo datos ETA desde: {archivo_eta_reciente}")

    try:
        df_eta = pd.read_csv(archivo_eta_reciente)
        
        if 'Omega_L0' in df_eta.columns:
            omega_val = df_eta['Omega_L0'].iloc[0]
            df_eta = df_eta[df_eta['Omega_L0'] == omega_val]
            
        df_eta = df_eta.sort_values(by='eta').reset_index(drop=True)
        
        eta = df_eta['eta'].values
        eta_10 = df_eta['eta'].values * 1e10

        pr = df_eta['p']
        Y_p_mass = 4 * df_eta['4He']
        D_H      = df_eta['d'] / df_eta['p']
        He3_H    = df_eta['3He'] / df_eta['p']
        Li7_H    = (df_eta['7Li'] + df_eta['7Be']) / df_eta['p']

        df_eta['Pesados'] = 0
        for el, A in mass_pesados.items():
            if el in df_eta.columns:
                df_eta['Pesados'] += A * df_eta[el]

        # Likelihood fitting
        
        Yp_obs, sigma_Yp = 0.245, 0.003
        D_obs, sigma_D = 2.527e-5, 0.03e-5    
        Li_obs, sigma_Li = 1.6e-10, 0.3e-10    
        n_sigma = 2.0
        
        log_eta = np.log10(eta)
        f_pr = interp1d(log_eta, pr, kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_Yp = interp1d(log_eta, Y_p_mass, kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_D  = interp1d(log_eta, D_H, kind='cubic', bounds_error=False, fill_value="extrapolate")
        f_Li = interp1d(log_eta, Li7_H, kind='cubic', bounds_error=False, fill_value="extrapolate")

        try:
            from scipy.optimize import brentq, minimize
            
            log_eta_inicio = np.min(log_eta)
            log_eta_fin = np.max(log_eta)
            
            def encontrar_eta_seguro_D(target):
                val_min = f_D(log_eta_inicio)
                val_max = f_D(log_eta_fin)
                
                if (val_min - target) * (val_max - target) <= 0:
                    return brentq(lambda x: f_D(x) - target, log_eta_inicio, log_eta_fin)
                else:
                    res = minimize(lambda x: np.abs(f_D(x[0]) - target), 
                                   x0=[(log_eta_inicio + log_eta_fin)/2], 
                                   bounds=[(log_eta_inicio, log_eta_fin)])
                    return res.x[0]

            log_eta_opt = encontrar_eta_seguro_D(D_obs)
            log_eta_min_val = encontrar_eta_seguro_D(D_obs - n_sigma * sigma_D)
            log_eta_max_val = encontrar_eta_seguro_D(D_obs + n_sigma * sigma_D)
            
            eta_opt = 10**log_eta_opt
            eta_min_val, eta_max_val = sorted([10**log_eta_min_val, 10**log_eta_max_val])
            ajuste_exitoso = True
            
            delta_eta_plus = eta_max_val - eta_opt
            delta_eta_minus = eta_opt - eta_min_val
            delta_eta_avg_2sigma = (delta_eta_plus + delta_eta_minus) / 2.0
            
            print("# ETA ÓPTIMO (Deuterio)")
            print(f"-> Eta óptimo hallado: {eta_opt:.4e} ± {delta_eta_avg_2sigma:.4e} (2 sigma)")
            print(f"[Rango exacto 2 sigma: {eta_min_val:.4e} a {eta_max_val:.4e}]")
            print("\nFracciones de masa resultantes:")
            print(f"  p (H) : {f_pr(np.log10(eta_opt)):.5e}")
            print(f"  Y_p (4He) : {f_Yp(np.log10(eta_opt)):.5e}")
            print(f"  D/H       : {f_D(np.log10(eta_opt)):.5e}")
            print(f"  7Li/H     : {f_Li(np.log10(eta_opt)):.5e}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Fallo en la optimización: {e}")
            ajuste_exitoso = False

        # LIKELIHOOD
        eta_high_res = np.logspace(np.log10(np.min(eta)), np.log10(np.max(eta)), 50000)
        log_eta_high = np.log10(eta_high_res)
        umbral_validez = 1e-4 

        Y_p_high_res = f_Yp(log_eta_high)
        lik_Yp = np.exp(-0.5 * ((Y_p_high_res - Yp_obs) / sigma_Yp)**2)
        if np.max(lik_Yp) > umbral_validez: lik_Yp /= np.max(lik_Yp)
        else: lik_Yp[:] = 0.0

        D_high_res = f_D(log_eta_high)
        lik_D = np.exp(-0.5 * ((D_high_res - D_obs) / sigma_D)**2)
        if np.max(lik_D) > umbral_validez: lik_D /= np.max(lik_D)
        else: lik_D[:] = 0.0

        Li_high_res = f_Li(log_eta_high)
        lik_Li = np.exp(-0.5 * ((Li_high_res - Li_obs) / sigma_Li)**2)
        if np.max(lik_Li) > umbral_validez: lik_Li /= np.max(lik_Li)
        else: lik_Li[:] = 0.0

        eta_lcdm = 6.12e-10
        eta_cmb_min = 5.8e-10
        eta_cmb_max = 6.3e-10
        
        nombre_base_eta = os.path.splitext(os.path.basename(archivo_eta_reciente))[0]

        
        # FRACCIONES DE MASA FINALES VS ETA
        plt.figure(4, figsize=(10, 6), dpi=200)
        plt.loglog(eta, 1 * df_eta['p'], label='p', color='blue', linewidth=lw)
        plt.loglog(eta, 2 * df_eta['d'], label='D', color='green', linewidth=lw)
        plt.loglog(eta, 3 * df_eta['t_iso'], label='T', color='orange', linewidth=lw)
        plt.loglog(eta, 3 * df_eta['3He'], label=r'$^3$He', color='red', linewidth=lw)
        plt.loglog(eta, 4 * df_eta['4He'], label=r'$^4$He', color='cyan', linewidth=lw)
        plt.loglog(eta, 7 * df_eta['7Li'] + 7 * df_eta['7Be'], label=r'$^7$Li', color='magenta', linewidth=lw)
        plt.loglog(eta, 8 * df_eta['8B'], label='$^8$B', color='gray', linewidth=lw)
        plt.loglog(eta, df_eta['Pesados'], label=r'A $\geq$ 9', color='purple', linestyle='-', linewidth=lw) 
        
        plt.xlabel(r'Baryon-to-photon ratio ($\eta$)', fontsize=16)
        plt.ylabel(r'Final mass fractions $X_i$', fontsize=16)
        plt.grid(True, which="both", alpha=0.4, linestyle='--')
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=14)
        plt.xlim(np.min(eta), np.max(eta))
        plt.ylim(1e-25, 2)
        plt.tight_layout()

        pdf_filename_eta_completa = f"grafica_completa_{nombre_base_eta}.pdf"
        plt.savefig(pdf_filename_eta_completa, format='pdf', bbox_inches='tight')
        print(f"Gráfica de abundancias completas guardada como: {pdf_filename_eta_completa}")
        
        # SCRHAMM PLOT
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(14, 8), num=5)
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
        
        cmb_min, cmb_max = 5.8, 6.6
        
        # --- PANEL 1: Y_p ---
        ax1.plot(eta_10, Y_p_mass, color='darkmagenta', lw=3)
        ax1.set_ylim(0.23, 0.26) 
        ax1.set_ylabel(r'Y$_{\rm P}$', fontsize=26, rotation=0, ha='right', va='center')
        ax1.yaxis.set_label_coords(-0.15, 0.5) # <-- X=-0.15 (izquierda), Y=0.5 (centro exacto)
        ax1.axhspan(Yp_obs - n_sigma*sigma_Yp, Yp_obs + n_sigma*sigma_Yp, color='yellow', alpha=0.5, zorder=1)
        
        # --- PANEL 2: D/H ---
        ax2.loglog(eta_10, D_H, color='mediumblue', lw=3)
        ax2.set_ylim(9e-6, 1e-3) 
        ax2.set_ylabel(r'D/H$|_{\rm p}$', fontsize=26, rotation=90, ha='left', va='center')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_label_coords(1.05, 0.3)  # <-- X=1.15 (derecha), Y=0.5 (centro exacto)
        ax2.axhspan(D_obs - n_sigma*sigma_D, D_obs + n_sigma*sigma_D, color='yellow', alpha=0.5, zorder=1)
        
        # --- PANEL 3: 3He/H ---
        ax3.loglog(eta_10, He3_H, color='firebrick', lw=3)
        ax3.set_ylim(1e-6, 1e-4) 
        ax3.set_ylabel(r'$^3$He/H$|_{\rm p}$', fontsize=26, rotation=90, ha='right', va='center')
        ax3.yaxis.set_label_coords(-0.2, 0.7)
        
        # --- PANEL 4: 7Li/H ---
        ax4.loglog(eta_10, Li7_H, color='limegreen', lw=3)
        ax4.set_ylim(1e-11, 1e-9) 
        ax4.set_ylabel(r'$^7$Li/H$|_{\rm p}$', fontsize=26, rotation=90, ha='left', va='center')
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.set_label_coords(1.05, 0.3)
        ax4.axhspan(Li_obs - n_sigma*sigma_Li, Li_obs + n_sigma*sigma_Li, color='yellow', alpha=0.5, zorder=1)
        
        ax2.yaxis.set_label_position("right")
        ax4.yaxis.set_label_position("right")
        
        def log_minor_formatter(x, pos):
            if x in [2, 3, 4, 6]:
                return f"{int(x)}"
            return ""
        
        for ax in [ax3, ax4]:
            ax.set_xscale('log')
            ax.set_xlim(eta_10.min(), eta_10.max())
            ax.set_xlabel(r'Baryon-to-photon ratio $\eta \times 10^{10}$', fontsize=24)
            
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.xaxis.set_minor_formatter(ticker.FuncFormatter(log_minor_formatter))
        
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.grid(True, which='major', linestyle='-', alpha=0.5)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)
            ax.tick_params(axis='y', direction='in', right=True, which='both', labelsize=20)
            ax.tick_params(axis='x', direction='in', top=True, which='both', labelsize=20)
            
            lbl_cmb = 'CMB (Planck)' if i == 0 else None
            ax.axvspan(cmb_min, cmb_max, color='cornflowerblue', alpha=0.3, label=lbl_cmb)
            
            lbl_lcdm = 'LCDM' if i == 0 else None
            ax.axvline(eta_lcdm * 1e10, color='gray', linestyle=':', linewidth=2, label=lbl_lcdm)
        
            if ajuste_exitoso:
                lbl_opt = 'Best-fit ($D/H$)' if i == 0 else None
                ax.axvline(eta_opt * 1e10, color='red', linestyle='-.', linewidth=2, label=lbl_opt)
                
                lbl_band = r'$2\sigma$ range ($\eta_{opt}$)' if i == 0 else None
                ax.axvspan(eta_min_val * 1e10, eta_max_val * 1e10, color='tomato', alpha=0.4, zorder=1, label=lbl_band)
        
        ax1.legend(loc='best', fontsize=18, framealpha=0.9, edgecolor='gray')
        fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.12, hspace=0.10, wspace=0.25)
        
        pdf_filename_eta_schramm = f"grafica_schramm_{nombre_base_eta}.pdf"
        plt.savefig(pdf_filename_eta_schramm, format='pdf', bbox_inches='tight')
        print(f"Gráfica Schramm guardada como: {pdf_filename_eta_schramm}")
        
        # GRÁFICA DE LA LIKELIHOOD
        plt.figure(6, figsize=(15, 6), dpi=200)
        corte_2sigma = np.exp(-0.5 * (2)**2)
    
        # Curvas principales
        plt.plot(eta_high_res * 1e10, lik_Yp, color='darkmagenta', lw=3, label=r'$\mathcal{L}(Y_p)$')
        plt.plot(eta_high_res * 1e10, lik_D, color='yellow', lw=3, label=r'$\mathcal{L}(D/H)$')
        plt.plot(eta_high_res * 1e10, lik_Li, color='limegreen', lw=3, label=r'$\mathcal{L}(^7Li/H)$')
        plt.axvline(eta_lcdm * 1e10, color='gray', linestyle=':', linewidth=2, label=fr'$\eta$ (Planck) = {eta_lcdm:.2e}')
        
        if ajuste_exitoso:
            plt.axvline(eta_opt * 1e10, color='red', linestyle='-.', linewidth=2, label=fr'Best-fit $\eta$ ($D/H$) = {eta_opt:.2e}')
            
        # Bandas de color condicionales
        if np.max(lik_D) > corte_2sigma:
            plt.fill_between(eta_high_res * 1e10, 0, 1.05, where=(lik_D >= corte_2sigma), color='red', alpha=0.3, label=r'$2\sigma$ range ($D/H$)')

        plt.xlabel(r'Baryon-to-photon ratio $\eta \times 10^{10}$', fontsize=25)
        plt.ylabel(r'Likelihood $\mathcal{L}$', fontsize=25)
        plt.grid(True, which="both", alpha=0.4, linestyle='--')
        plt.tick_params(axis='both', which='major', labelsize=20, direction='in', top=True, right=True)
        plt.tick_params(axis='x', which='minor', labelsize=20, direction='in', top=True)
        
        plt.xscale('log')
        plt.xlim(eta_10.min(), eta_10.max())
        plt.ylim(0, 1.05)
        
        def log_minor_formatter(x, pos):
            if x in [4, 5, 6, 7]:
                return f"{int(x)}"
            return ""
    
        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(log_minor_formatter))
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=18, ncol=1)
        plt.tight_layout()

        # Guardado de la imagen
        pdf_filename_eta_lik = f"grafica_likelihood_multiple_{nombre_base_eta}.pdf"
        plt.savefig(pdf_filename_eta_lik, format='pdf', bbox_inches='tight')
        print(f"Gráfica de Likelihood guardada como: {pdf_filename_eta_lik}\n")
        
        
    except Exception as e:
        import traceback
        print(f"Error al procesar el archivo {archivo_eta_reciente}: {e}.")
        traceback.print_exc()
        print("Omitiendo Parte 2.")
        
# Mostrar todo al final
plt.show()
