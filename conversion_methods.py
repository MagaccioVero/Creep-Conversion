"""
Moduli di conversione Creep -> Moduli Dinamici
3 Metodi: NLREG, Schwarzl-Staverman, Spettrale
"""

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d
from scipy.integrate import quad


class NLREGConverter:
    """Metodo 1: NLREG - Minimizzazione dell'errore relativo con derivata seconda"""
    
    def __init__(self, t_raw, J_raw, N_elements=100):
        self.t_raw = t_raw
        self.J_raw = J_raw
        self.N_elements = N_elements
        self.tau_k = np.logspace(np.log10(t_raw.min())-1, np.log10(t_raw.max())+1, N_elements)
        self.d_ln_tau = np.log(self.tau_k[1] / self.tau_k[0])
        
    def find_optimal_lambda(self, lambda_min=-6, lambda_max=2, n_lambdas=80):
        """Ricerca del λ ottimale usando GCV"""
        lambdas = np.logspace(lambda_min, lambda_max, n_lambdas)
        gcv_scores = []
        
        K_weighted = self._get_weighted_kernel()
        K_T_K = K_weighted.T @ K_weighted
        L_T_L = self._get_regularization_matrix().T @ self._get_regularization_matrix()
        n_pts = len(self.J_raw)
        
        for lam in lambdas:
            K_reg = np.vstack([K_weighted, lam * self._get_regularization_matrix()])
            J_reg = np.concatenate([self.J_raw * (1.0 / self.J_raw), np.zeros(self.N_elements + 2)])
            
            res_tmp = lsq_linear(K_reg, J_reg, bounds=(0, np.inf))
            res_sq = np.linalg.norm(K_weighted @ res_tmp.x - (self.J_raw * (1.0 / self.J_raw)))**2
            
            try:
                inv_term = np.linalg.pinv(K_T_K + (lam**2) * L_T_L)
                trace_H = np.trace(inv_term @ K_T_K)
                den = (1.0 - trace_H / n_pts)**2
                gcv = res_sq / den if den > 0 else np.inf
            except:
                gcv = np.inf
            
            gcv_scores.append(gcv)
        
        best_lam = lambdas[np.argmin(gcv_scores)]
        return best_lam, np.log10(best_lam)
    
    def _get_kernel_matrix(self):
        """Matrice kernel per NLREG"""
        K_tau = (1.0 - np.exp(-np.outer(self.t_raw, 1.0/self.tau_k))) * self.d_ln_tau
        K_Jg = np.ones((len(self.t_raw), 1))
        t_max = self.t_raw.max()
        K_visc = (self.t_raw / t_max).reshape(-1, 1)
        return np.hstack([K_Jg, K_visc, K_tau])
    
    def _get_weighted_kernel(self):
        """Kernel ponderato per errore relativo"""
        K = self._get_kernel_matrix()
        W = 1.0 / self.J_raw
        return K * W[:, np.newaxis]
    
    def _get_regularization_matrix(self):
        """Matrice di regolarizzazione (derivata seconda)"""
        L_matrix = np.zeros((self.N_elements + 2, self.N_elements + 2))
        D2 = np.eye(self.N_elements) * 2 - np.diag(np.ones(self.N_elements-1), 1) - np.diag(np.ones(self.N_elements-1), -1)
        D2[0, 0] = 1
        D2[-1, -1] = 1
        L_matrix[2:, 2:] = D2
        return L_matrix
    
    def convert(self, lambda_value=0.01):
        """Conversione NLREG con lambda specificato"""
        K_weighted = self._get_weighted_kernel()
        J_weighted = self.J_raw * (1.0 / self.J_raw)
        L_matrix = self._get_regularization_matrix()
        
        K_reg = np.vstack([K_weighted, lambda_value * L_matrix])
        J_reg = np.concatenate([J_weighted, np.zeros(self.N_elements + 2)])
        
        res = lsq_linear(K_reg, J_reg, bounds=(0, np.inf))
        
        J_g = res.x[0]
        t_max = self.t_raw.max()
        phi_0 = res.x[1] / t_max
        L_tau = res.x[2:]
        
        omega_target = 1.0 / self.t_raw
        
        J_prime = np.ones_like(omega_target) * J_g
        J_double = phi_0 / omega_target
        
        for k in range(self.N_elements):
            denominatore = 1.0 + (omega_target * self.tau_k[k])**2
            J_prime += L_tau[k] * self.d_ln_tau / denominatore
            J_double += L_tau[k] * self.d_ln_tau * (omega_target * self.tau_k[k]) / denominatore
        
        abs_J_sq = J_prime**2 + J_double**2
        G_prime = J_prime / abs_J_sq
        G_double_prime = J_double / abs_J_sq
        
        return pd.DataFrame({
            'Frequenza w [rad/s]': omega_target,
            "G' [Pa]": G_prime,
            "G'' [Pa]": G_double_prime
        })


class SchwarzlStavermanConverter:
    """Metodo 2: Schwarzl e Staverman - Approssimazione analitica"""
    
    def __init__(self, t_raw, J_raw):
        self.t_raw = t_raw
        self.J_raw = J_raw
        self.J_inf = J_raw[-1]  # Complianza a lungo tempo
        self.J_0 = J_raw[0]      # Complianza istantanea (approssimata)
        
    def convert(self):
        """
        Metodo di Schwarzl-Staverman:
        Fornisce relazioni analitiche approssimate tra J(t) e G'(ω), G''(ω)
        """
        omega = 1.0 / self.t_raw
        
        # Calcolo della derivata prima di J(t) rispetto al tempo
        dJ_dt = np.gradient(self.J_raw, self.t_raw)
        
        # Derivata rispetto a ln(t): dJ/dln(t) = t * dJ/dt
        dJ_dln_t = self.t_raw * dJ_dt
        
        # Approssimazione di Schwarzl-Staverman per i moduli di compliance
        J_prime = self.J_raw - 0.86 * dJ_dln_t
        J_double_prime = (np.pi / 2.0) * dJ_dln_t
        
        # Assicura positività fisica
        J_prime = np.maximum(J_prime, 1e-12)
        J_double_prime = np.maximum(J_double_prime, 1e-12)
        
        # Conversione dai moduli di compliance ai moduli dinamici G' e G''
        abs_J_sq = J_prime**2 + J_double_prime**2
        G_prime = J_prime / abs_J_sq
        G_double_prime = J_double_prime / abs_J_sq
        
        return pd.DataFrame({
            'Frequenza w [rad/s]': omega,
            "G' [Pa]": G_prime,
            "G'' [Pa]": G_double_prime
        })


class SpectralConverter:
    """Metodo 3: Basato sugli Spettri - Analisi dello spettro di rilassamento"""
    
    def __init__(self, t_raw, J_raw, n_kernels=50):
        self.t_raw = t_raw
        self.J_raw = J_raw
        self.n_kernels = n_kernels
        self.tau_spectrum = np.logspace(np.log10(t_raw.min())-0.5, np.log10(t_raw.max())+0.5, n_kernels)
        
    def _estimate_spectrum(self):
        """Stima dello spettro di rilassamento L(τ) dal creep"""
        # Costruisce la matrice kernel per lo spettro
        K_spectrum = np.zeros((len(self.t_raw), self.n_kernels))
        
        for i, tau in enumerate(self.tau_spectrum):
            # Kernel di ritardo: (1 - exp(-t/τ))
            K_spectrum[:, i] = 1.0 - np.exp(-self.t_raw / tau)
        
        # Regolarizzazione di Tikhonov per stimare lo spettro
        alpha = 0.001  # Parametro di regolarizzazione ridotto
        
        # Risolve: (K'K + α*I)L = K'*(J - J0)
        K_T_K = K_spectrum.T @ K_spectrum
        K_T_J = K_spectrum.T @ (self.J_raw - self.J_raw[0])
        
        L_spectrum = np.linalg.solve(K_T_K + alpha * np.eye(self.n_kernels), K_T_J)
        L_spectrum = np.maximum(L_spectrum, 0)  # Forza positività
        
        return L_spectrum
    
    def convert(self):
        """Conversione usando spettro di ritardo"""
        L_spectrum = self._estimate_spectrum()
        
        # Frequenze target
        omega = 1.0 / self.t_raw
        
        # Calcolo complianze usando spettro
        J_prime = np.zeros_like(omega, dtype=float)
        J_double_prime = np.zeros_like(omega, dtype=float)
        
        d_ln_tau = np.log(self.tau_spectrum[1] / self.tau_spectrum[0]) if len(self.tau_spectrum) > 1 else 1.0
        
        # Componente elastica (J_0)
        J_0 = self.J_raw[0]
        
        for i, om in enumerate(omega):
            for j, tau in enumerate(self.tau_spectrum):
                denominatore = 1.0 + (om * tau)**2
                J_prime[i] += L_spectrum[j] * d_ln_tau / denominatore
                J_double_prime[i] += L_spectrum[j] * d_ln_tau * (om * tau) / denominatore
            
            # Aggiunge il contributo elastico
            J_prime[i] += J_0
        
        # Assicura positività
        J_prime = np.maximum(J_prime, 1e-12)
        J_double_prime = np.maximum(J_double_prime, 1e-12)
        
        abs_J_sq = J_prime**2 + J_double_prime**2
        G_prime = J_prime / abs_J_sq
        G_double_prime = J_double_prime / abs_J_sq
        
        return pd.DataFrame({
            'Frequenza w [rad/s]': omega,
            "G' [Pa]": G_prime,
            "G'' [Pa]": G_double_prime
        })
