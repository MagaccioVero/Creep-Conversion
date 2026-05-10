"""
Utilità per caricamento e pre-elaborazione dei dati
"""

import pandas as pd
import numpy as np
import os


class DataLoader:
    """Gestione caricamento e pre-elaborazione dati"""
    
    @staticmethod
    def load_file(filepath, skiprows=0, encoding='utf-8'):
        """Carica file CSV, TXT, DAT, o Excel"""
        try:
            if filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath, skiprows=skiprows)
            elif filepath.endswith(('.csv', '.txt', '.dat')):
                try:
                    df = pd.read_csv(filepath, sep=None, engine='python', encoding=encoding, skiprows=skiprows)
                except UnicodeDecodeError:
                    # Prova con encoding diversi
                    for enc in ['utf-16', 'latin1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(filepath, sep=None, engine='python', encoding=enc, skiprows=skiprows)
                            break
                        except:
                            continue
            else:
                raise ValueError("Formato file non supportato")
            
            return df
        
        except Exception as e:
            raise Exception(f"Errore nel caricamento file: {e}")
    
    @staticmethod
    def prepare_creep_data(df, col_tempo, col_J, col_strain=None, col_stress=None):
        """
        Pre-elabora i dati di creep
        
        Args:
            df: DataFrame caricato
            col_tempo: Nome colonna tempo
            col_J: Nome colonna compliance (se fornita direttamente)
            col_strain: Nome colonna strain (se J deve essere calcolata)
            col_stress: Nome colonna stress (se J deve essere calcolata)
        
        Returns:
            DataFrame elaborato con colonne 'Tempo' e 'J'
        """
        
        # Conversione colonne a numeriche
        cols_to_convert = [col_tempo]
        if col_J:
            cols_to_convert.append(col_J)
        if col_strain:
            cols_to_convert.append(col_strain)
        if col_stress:
            cols_to_convert.append(col_stress)
        
        for col in cols_to_convert:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calcola J se necessario
        if col_strain is not None and col_stress is not None:
            sigma_0 = df[col_stress].max()
            df['J'] = df[col_strain] / sigma_0
            sigma_0_val = sigma_0
        else:
            df['J'] = df[col_J]
            sigma_0_val = None
        
        # Rimuove valori NaN
        df = df[[col_tempo, 'J']].dropna()
        
        # Rimuove punti con tempo <= 0
        df = df[df[col_tempo] > 0]
        
        # Ordina per tempo e rimuove duplicati
        df = df.sort_values(by=col_tempo)
        df = df.drop_duplicates(subset=[col_tempo])
        
        # Rinomina colonne
        df_out = pd.DataFrame({
            'Tempo': df[col_tempo].values,
            'J': df['J'].values
        })
        
        return df_out, sigma_0_val
    
    @staticmethod
    def clean_moduli_data(df_moduli):
        """Pulisce i dati dei moduli (rimuove valori non positivi)"""
        df_moduli = df_moduli.copy()
        
        # Rimuove righe con G' o G'' non positivi
        df_moduli = df_moduli[(df_moduli["G' [Pa]"] > 0) & (df_moduli["G'' [Pa]"] > 0)]
        
        # Arrotonda a 2 cifre decimali in notazione scientifica
        for col in df_moduli.columns:
            if col != 'Frequenza w [rad/s]':
                df_moduli[col] = df_moduli[col].apply(lambda x: float(f"{x:.2e}"))
        
        return df_moduli
    
    @staticmethod
    def export_to_txt(df_moduli, output_path):
        """Esporta dati in formato txt con tabulazione"""
        txt_data = df_moduli.to_csv(index=False, float_format="%.2e", decimal=",", sep="\t")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(txt_data)
