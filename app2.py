import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from conversion_methods import NLREGConverter, SchwarzlStavermanConverter, SpectralConverter
from data_utils import DataLoader

# --- 1. CONFIGURAZIONE PAGINA E MEMORIA ---
st.set_page_config(page_title="App Reologia Avanzata", page_icon="🧪", layout="wide")

if 'files_data' not in st.session_state:
    st.session_state.files_data = {}
if 'file_list' not in st.session_state:
    st.session_state.file_list = []

st.title("🧪 Conversione Creep -> G' & G'' by Mago_Vero")
st.write("Motore di conversione con 3 metodi: NLREG, Approssimazione di Schwarzl e Staverman, e Metodo basato sugli Spettri.")

# --- 2. CARICAMENTO MULTIPLI FILE ---
st.subheader("1. Caricamento Dati J(t) - Seleziona più file")

uploaded_files = st.file_uploader(
    "Carica i file (CSV, TXT, DAT, Excel)", 
    type=["csv", "xlsx", "txt", "dat"],
    accept_multiple_files=True
)

# Elabora i file caricati e aggiungili alla session_state
if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    
    # Rimuovi i file che non sono più caricati
    files_to_remove = [key for key in st.session_state.file_list if key not in current_file_names]
    for file_key in files_to_remove:
        if file_key in st.session_state.files_data:
            del st.session_state.files_data[file_key]
        st.session_state.file_list.remove(file_key)
    
    # Aggiungi i nuovi file
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        
        if file_key not in st.session_state.files_data:
            st.session_state.files_data[file_key] = {
                'file': uploaded_file,
                'righe_da_saltare': DataLoader.detect_skiprows(uploaded_file),
                'df': None,
                'dati_pronti': False,
                'col_tempo': None,
                'col_complianza': None,
                'col_strain': None,
                'col_stress': None,
                'tipo_input': None,
                'df_creep': None,
                'sigma_0': None,
                'results': {},
                'df_agg': None,
                'cols_agg': None,
                'lambda_ottimale': -2.0,
                'need_lambda_recalc': False
            }
        
        if file_key not in st.session_state.file_list:
            st.session_state.file_list.append(file_key)
else:
    # Se nessun file è caricato, svuota la lista
    st.session_state.file_list = []
    st.session_state.files_data = {}

# --- 3. CREAZIONE TAB PER OGNI FILE ---
if st.session_state.file_list:
    st.write("---")
    tabs = st.tabs([f"📄 {fname}" for fname in st.session_state.file_list])
    
    for tab_idx, tab in enumerate(tabs):
        file_key = st.session_state.file_list[tab_idx]
        file_data = st.session_state.files_data[file_key]
        
        with tab:
            st.subheader(f"Elaborazione: {file_key}")
            
            # Input per righe da saltare specifico del file
            righe_input = st.number_input(
                f"Righe iniziali da saltare:",
                min_value=0,
                value=int(file_data['righe_da_saltare']),
                step=1,
                key=f"righe_{file_key}"
            )
            
            # Se l'utente cambia il numero di righe, ricarichiamo il file
            if righe_input != file_data['righe_da_saltare']:
                file_data['righe_da_saltare'] = righe_input
                file_data['df'] = None  # Reset del dataframe per ricaricarlo
                st.rerun()
            
            try:
                # Carica il file se non è già caricato
                if file_data['df'] is None:
                    uploaded_file = file_data['file']
                    uploaded_file.seek(0)
                    
                    if uploaded_file.name.endswith(('.csv', '.txt', '.dat')):
                        try:
                            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8', skiprows=file_data['righe_da_saltare'])
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            try:
                                df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-16', skiprows=file_data['righe_da_saltare'])
                            except UnicodeDecodeError:
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1', skiprows=file_data['righe_da_saltare'])
                    else:
                        df = pd.read_excel(uploaded_file, skiprows=file_data['righe_da_saltare'])
                    
                    file_data['df'] = df
                else:
                    df = file_data['df']
                
                with st.expander("📋 Visualizza dati grezzi"):
                    st.dataframe(df.head(10))
                
                # --- SELEZIONE COLONNE ---
                st.write("### ⚙️ Elaborazione Dati e Calcolo Compliance")
                colonne = df.columns.tolist()
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    tipo_input = st.radio(
                        f"Input per {file_key}:", 
                        ["Ho già la colonna della Compliance J(t)", "Calcola da Strain e Stress"],
                        horizontal=False,
                        key=f"tipo_{file_key}"
                    )
                    file_data['tipo_input'] = tipo_input
                
                with col_right:
                    if tipo_input == "Calcola da Strain e Stress":
                        col_tempo = st.selectbox("Colonna Tempo [s]:", colonne, index=0, key=f"tempo_{file_key}")
                        col_strain = st.selectbox("Colonna Strain γ:", colonne, index=1 if len(colonne)>1 else 0, key=f"strain_{file_key}")
                        col_stress = st.selectbox("Colonna Stress σ [Pa]:", colonne, index=2 if len(colonne)>2 else 0, key=f"stress_{file_key}")
                        file_data['col_tempo'] = col_tempo
                        file_data['col_strain'] = col_strain
                        file_data['col_stress'] = col_stress
                    else:
                        col_tempo = st.selectbox("Colonna Tempo [s]:", colonne, index=0, key=f"tempo_{file_key}")
                        col_complianza = st.selectbox("Colonna Compliance J(t) [1/Pa]:", colonne, index=1 if len(colonne)>1 else 0, key=f"comp_{file_key}")
                        file_data['col_tempo'] = col_tempo
                        file_data['col_complianza'] = col_complianza
                
                # Bottone per elaborare dati e fare conversione automatica
                if st.button(f"▶ Elabora Dati e Converti - {file_key}", key=f"btn_elabora_{file_key}", type="primary"):
                    try:
                        if tipo_input == "Calcola da Strain e Stress":
                            df_creep, sigma_0_val = DataLoader.prepare_creep_data(df.copy(), col_tempo, None, col_strain, col_stress)
                        else:
                            df_creep, sigma_0_val = DataLoader.prepare_creep_data(df.copy(), col_tempo, col_complianza)
                        
                        # Verifica se i dati sono vuoti
                        if df_creep.empty or len(df_creep) == 0:
                            st.error("❌ Nessun dato valido trovato dopo la pre-elaborazione. Verifica i dati di input (possibili cause: valori NaN, tempi ≤ 0, colonne errate).")
                            continue
                        
                        file_data['df_creep'] = df_creep
                        file_data['sigma_0'] = sigma_0_val
                        file_data['dati_pronti'] = True
                        
                        # Esecuzione automatica della conversione
                        t_raw = df_creep['Tempo'].values
                        J_raw = df_creep['J'].values
                        
                        with st.spinner("🔄 Ricerca λ ottimale (GCV)..."):
                            converter_nlreg = NLREGConverter(t_raw, J_raw, N_elements=100)
                            best_lambda, log_lambda = converter_nlreg.find_optimal_lambda()
                            file_data['lambda_ottimale'] = log_lambda
                        
                        lambda_reg = 10 ** log_lambda
                        
                        with st.spinner("Esecuzione conversione con 3 metodi..."):
                            results = {}
                            
                            # 1. NLREG
                            converter_nlreg = NLREGConverter(t_raw, J_raw, N_elements=100)
                            df_nlreg = converter_nlreg.convert(lambda_reg)
                            results["NLREG"] = df_nlreg
                            
                            # 2. Schwarzl-Staverman
                            converter_sch = SchwarzlStavermanConverter(t_raw, J_raw)
                            df_sch = converter_sch.convert()
                            results["Schwarzl"] = df_sch
                            
                            # 3. Spettrale
                            converter_spec = SpectralConverter(t_raw, J_raw, n_kernels=50)
                            df_spec = converter_spec.convert()
                            results["Spettrale"] = df_spec
                            
                            file_data['results'] = results
                        
                        st.success(f"✅ Dati elaborati ({len(df_creep)} punti) e conversione completata!")
                    except Exception as e:
                        st.error(f"❌ Errore: {e}")
                
                # Se dati pronti, mostra parametri aggiustabili
                if file_data['dati_pronti']:
                    st.write("---")
                    st.write("### 🔧 Parametri di Conversione")
                    
                    df_creep = file_data['df_creep']
                    t_raw = df_creep['Tempo'].values
                    J_raw = df_creep['J'].values
                    
                    col_lambda, col_btn_reconvert = st.columns([3, 1])
                    with col_lambda:
                        log_lambda = st.slider(
                            "λ (Logaritmico):", 
                            min_value=-6.0, max_value=2.0, 
                            value=float(file_data['lambda_ottimale']),
                            step=0.1,
                            key=f"lambda_{file_key}"
                        )
                    
                    lambda_reg = 10 ** log_lambda
                    st.write(f"**λ applicato:** {lambda_reg:.2e}")
                    
                    if st.button(f"🔄 Riconverti con nuovo λ - {file_key}", key=f"btn_reconvert_{file_key}"):
                        # Verifica se i dati sono vuoti
                        if df_creep.empty or len(df_creep) == 0:
                            st.error("❌ Nessun dato valido trovato. Rielabora i dati prima.")
                            st.stop()
                        
                        with st.spinner("Riconversione con nuovo λ..."):
                            results = {}
                            
                            # NLREG
                            converter_nlreg = NLREGConverter(t_raw, J_raw, N_elements=100)
                            df_nlreg = converter_nlreg.convert(lambda_reg)
                            results["NLREG"] = df_nlreg
                            
                            # Schwarzl-Staverman
                            converter_sch = SchwarzlStavermanConverter(t_raw, J_raw)
                            df_sch = converter_sch.convert()
                            results["Schwarzl"] = df_sch
                            
                            # Spettrale
                            converter_spec = SpectralConverter(t_raw, J_raw, n_kernels=50)
                            df_spec = converter_spec.convert()
                            results["Spettrale"] = df_spec
                            
                            file_data['results'] = results
                            st.success("✅ Riconversione completata!")
                    
                    # Visualizzazione risultati
                    if file_data['results']:
                        st.write("---")
                        st.write("### 📈 Risultati Conversione")
                        
                        # Sezione dati aggiuntivi
                        st.write("#### Sovrapposizione Frequency sweep")
                        
                        uploaded_file_agg = st.file_uploader(
                            "Carica dati aggiuntivi (w, G', G'')",
                            type=["csv", "xlsx", "txt", "dat"],
                            key=f"file_agg_{file_key}"
                        )
                        
                        if uploaded_file_agg is not None:
                            default_skip_agg = DataLoader.detect_skiprows(uploaded_file_agg)
                            righe_agg = st.number_input("Righe da saltare:", min_value=0, value=int(default_skip_agg), step=1, key=f"righe_agg_{file_key}")
                            
                            try:
                                if uploaded_file_agg.name.endswith(('.csv', '.txt', '.dat')):
                                    try:
                                        df_agg_raw = pd.read_csv(uploaded_file_agg, sep=None, engine='python', encoding='utf-8', skiprows=righe_agg)
                                    except UnicodeDecodeError:
                                        uploaded_file_agg.seek(0)
                                        try:
                                            df_agg_raw = pd.read_csv(uploaded_file_agg, sep=None, engine='python', encoding='utf-16', skiprows=righe_agg)
                                        except UnicodeDecodeError:
                                            uploaded_file_agg.seek(0)
                                            df_agg_raw = pd.read_csv(uploaded_file_agg, sep=None, engine='python', encoding='latin1', skiprows=righe_agg)
                                else:
                                    df_agg_raw = pd.read_excel(uploaded_file_agg, skiprows=righe_agg)
                                
                                colonne_agg = df_agg_raw.columns.tolist()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1: 
                                    col_w_agg = st.selectbox("Frequenza w:", colonne_agg, index=0, key=f"w_agg_{file_key}")
                                with col2: 
                                    col_Gp_agg = st.selectbox("G' [Pa]:", colonne_agg, index=1 if len(colonne_agg)>1 else 0, key=f"gp_agg_{file_key}")
                                with col3: 
                                    col_Gpp_agg = st.selectbox("G'' [Pa]:", colonne_agg, index=2 if len(colonne_agg)>2 else 0, key=f"gpp_agg_{file_key}")
                                
                                if st.button(f"✅ Sovrapponi Dati - {file_key}", key=f"btn_sovrapp_{file_key}"):
                                    df_agg = df_agg_raw.copy()
                                    for col in [col_w_agg, col_Gp_agg, col_Gpp_agg]:
                                        if df_agg[col].dtype == object: 
                                            df_agg[col] = df_agg[col].astype(str).str.replace(',', '.')
                                        df_agg[col] = pd.to_numeric(df_agg[col], errors='coerce')
                                    
                                    file_data['df_agg'] = df_agg.dropna(subset=[col_w_agg, col_Gp_agg, col_Gpp_agg])
                                    file_data['cols_agg'] = (col_w_agg, col_Gp_agg, col_Gpp_agg)
                                    st.success("✅ Dati aggiuntivi pronti!")
                            except Exception as e:
                                st.error(f"❌ Errore: {e}")
                        
                        # Funzione per plot
                        def plot_moduli(df_moduli, title, file_key):
                            df_moduli = DataLoader.clean_moduli_data(df_moduli)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_moduli['Frequenza w [rad/s]'], y=df_moduli["G' [Pa]"], mode='markers', name="G' (Storage)", marker=dict(size=8, symbol='diamond', color='red')))
                            fig.add_trace(go.Scatter(x=df_moduli['Frequenza w [rad/s]'], y=df_moduli["G'' [Pa]"], mode='markers', name="G'' (Loss)", marker=dict(size=8, symbol='square', color='blue')))
                            
                            if file_data['df_agg'] is not None:
                                df_agg = file_data['df_agg']
                                col_w, col_Gp, col_Gpp = file_data['cols_agg']
                                fig.add_trace(go.Scatter(x=df_agg[col_w], y=df_agg[col_Gp], mode='lines', name="G' (frequency)", line=dict(color='lightcoral', dash='dash')))
                                fig.add_trace(go.Scatter(x=df_agg[col_w], y=df_agg[col_Gpp], mode='lines', name="G'' (frequency)", line=dict(color='lightskyblue', dash='dash')))
                            
                            fig.update_layout(
                                title=title,
                                xaxis_title="Frequenza w [rad/s]",
                                yaxis_title="Modulo [Pa]",
                                template="plotly_white",
                                hovermode="x unified",
                                width=800,
                                height=600
                            )
                            fig.update_xaxes(type="log", exponentformat="power", dtick=1)
                            fig.update_yaxes(type="log", exponentformat="power", dtick=1, minor=dict(ticks="inside", showgrid=True))
                            return fig
                        
                        # Mostra i 3 grafici con chiavi univoche
                        col_plot1, col_plot2 = st.columns(2)
                        with col_plot1:
                            st.plotly_chart(plot_moduli(file_data['results']["NLREG"], "NLREG", file_key), use_container_width=True, key=f"plot_nlreg_{file_key}")
                            st.plotly_chart(plot_moduli(file_data['results']["Schwarzl"], "Schwarzl-Staverman", file_key), use_container_width=True, key=f"plot_schwarzl_{file_key}")
                        
                        with col_plot2:
                            st.plotly_chart(plot_moduli(file_data['results']["Spettrale"], "Spettrale", file_key), use_container_width=True, key=f"plot_spectral_{file_key}")
                        
                        # Esportazione
                        st.write("---")
                        st.write("### 📊 Esportazione Risultati per Metodo")
                        
                        nome_input = os.path.splitext(file_key)[0]
                        
                        # Preparazione dati per ogni metodo
                        df_nlreg_export = file_data['results']["NLREG"].copy()
                        df_schwarzl_export = file_data['results']["Schwarzl"].copy()
                        df_spettrale_export = file_data['results']["Spettrale"].copy()
                        
                        # Creazione dei 3 bottoni di download
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            txt_nlreg = df_nlreg_export.to_csv(index=False, float_format="%.2e", decimal=",", sep="\t").encode('utf-8')
                            file_output_nlreg = f'{nome_input}_NLREG.txt'
                            st.download_button(
                                label="📥 Scarica NLREG",
                                data=txt_nlreg,
                                file_name=file_output_nlreg,
                                mime='text/plain',
                                key=f"btn_download_nlreg_{file_key}"
                            )
                        
                        with col_btn2:
                            txt_schwarzl = df_schwarzl_export.to_csv(index=False, float_format="%.2e", decimal=",", sep="\t").encode('utf-8')
                            file_output_schwarzl = f'{nome_input}_Schwarzl.txt'
                            st.download_button(
                                label="📥 Scarica Schwarzl",
                                data=txt_schwarzl,
                                file_name=file_output_schwarzl,
                                mime='text/plain',
                                key=f"btn_download_schwarzl_{file_key}"
                            )
                        
                        with col_btn3:
                            txt_spettrale = df_spettrale_export.to_csv(index=False, float_format="%.2e", decimal=",", sep="\t").encode('utf-8')
                            file_output_spettrale = f'{nome_input}_Spettrale.txt'
                            st.download_button(
                                label="📥 Scarica Spettrale",
                                data=txt_spettrale,
                                file_name=file_output_spettrale,
                                mime='text/plain',
                                key=f"btn_download_spettrale_{file_key}"
                            )
            
            except Exception as e:
                st.error(f"❌ Errore elaborazione {file_key}: {e}")
else:
    st.info("📤 Carica i file per iniziare")

