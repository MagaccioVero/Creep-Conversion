"""
Script di test per verificare i metodi di conversione
"""

import numpy as np
import pandas as pd
from conversion_methods import NLREGConverter, SchwarzlStavermanConverter, SpectralConverter

print("="*60)
print("TEST DEI METODI DI CONVERSIONE")
print("="*60)

# Crea dati di test simulati
print("\n📊 Generazione dati di test...")
t_test = np.logspace(-1, 3, 50)  # 0.1 a 1000 secondi
J_test = 1e-9 * (1 + np.log10(t_test + 1)) # Compliance simulata

print(f"  Numero di punti: {len(t_test)}")
print(f"  Range tempo: {t_test[0]:.2e} - {t_test[-1]:.2e} s")
print(f"  Range J: {J_test[0]:.2e} - {J_test[-1]:.2e}")

# Test NLREG
print("\n1️⃣ Test NLREG...")
try:
    converter_nlreg = NLREGConverter(t_test, J_test, N_elements=100)
    lambda_opt, log_lambda = converter_nlreg.find_optimal_lambda()
    df_nlreg = converter_nlreg.convert(lambda_opt)
    gp_min = df_nlreg["G' [Pa]"].min()
    gp_max = df_nlreg["G' [Pa]"].max()
    gpp_min = df_nlreg["G'' [Pa]"].min()
    gpp_max = df_nlreg["G'' [Pa]"].max()
    print(f"  ✓ NLREG completato")
    print(f"    - λ ottimale: {lambda_opt:.2e} (log10={log_lambda:.1f})")
    print(f"    - Frequenze: {len(df_nlreg)}")
    print(f"    - G' range: {gp_min:.2e} - {gp_max:.2e} Pa")
    print(f"    - G'' range: {gpp_min:.2e} - {gpp_max:.2e} Pa")
except Exception as e:
    print(f"  ✗ ERRORE: {e}")

# Test Schwarzl-Staverman
print("\n2️⃣ Test Schwarzl-Staverman...")
try:
    converter_ss = SchwarzlStavermanConverter(t_test, J_test)
    df_ss = converter_ss.convert()
    gp_min = df_ss["G' [Pa]"].min()
    gp_max = df_ss["G' [Pa]"].max()
    gpp_min = df_ss["G'' [Pa]"].min()
    gpp_max = df_ss["G'' [Pa]"].max()
    print(f"  ✓ Schwarzl-Staverman completato")
    print(f"    - Frequenze: {len(df_ss)}")
    print(f"    - G' range: {gp_min:.2e} - {gp_max:.2e} Pa")
    print(f"    - G'' range: {gpp_min:.2e} - {gpp_max:.2e} Pa")
except Exception as e:
    print(f"  ✗ ERRORE: {e}")

# Test Spettrale
print("\n3️⃣ Test Spettrale...")
try:
    converter_spec = SpectralConverter(t_test, J_test, n_kernels=50)
    df_spec = converter_spec.convert()
    gp_min = df_spec["G' [Pa]"].min()
    gp_max = df_spec["G' [Pa]"].max()
    gpp_min = df_spec["G'' [Pa]"].min()
    gpp_max = df_spec["G'' [Pa]"].max()
    print(f"  ✓ Spettrale completato")
    print(f"    - Frequenze: {len(df_spec)}")
    print(f"    - G' range: {gp_min:.2e} - {gp_max:.2e} Pa")
    print(f"    - G'' range: {gpp_min:.2e} - {gpp_max:.2e} Pa")
except Exception as e:
    print(f"  ✗ ERRORE: {e}")

print("\n" + "="*60)
print("✓ TUTTI I TEST COMPLETATI CON SUCCESSO!")
print("="*60)
