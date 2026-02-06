''' Codigo creado por: Benjamin Fuentes Valdebenito '''

import streamlit as st
import pandas as pd
import pyalex
from pyalex import Authors, Works
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import unicodedata
import matplotlib.pyplot as plt
from difflib import SequenceMatcher 

# Configuration
pyalex.config.email = "benjapoitiers@gmail.com"
ANID_WEIGHT_MULTIPLIER = 6 
EURO_TO_CLP_RATE = 1012 # Tasa de conversión estimada para Euros
DOLAR_TO_CLP_RATE = 856 # Tasa de conversión estimada para Dolares

# --- FILE NAMES ---
FILE_PAPERS_VIGENTES = "Papers_Vigentes_Reparado_20260128_104709.xlsx" 
FILE_EMAILS_ADMIN = "ACAD-DOC_PUBLICO.xlsx"
FILE_ANID_HISTORIC = "BDH_HISTORICA.xlsx"
FILE_ORCID_USM = "data_orcid_utfm.xlsx" 

# --- HELPER FUNCTIONS ---
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return " ".join(text.split())

def reconstruct_abstract(inverted_index):
    if not inverted_index: return ""
    try:
        max_index = max([max(positions) for positions in inverted_index.values()])
        abstract_list = [""] * (max_index + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                abstract_list[pos] = word
        return " ".join(abstract_list)
    except: return ""

def calculate_total_clp(df_projects):
    """Calcula el total en CLP convirtiendo Euros y Miles de Pesos."""
    if df_projects.empty or 'MONTO_ADJUDICADO' not in df_projects.columns or 'MONEDA' not in df_projects.columns:
        return 0
    
    total_clp = 0
    for _, row in df_projects.iterrows():
        try:
            monto = float(row['MONTO_ADJUDICADO'])
            if pd.isna(monto): continue
            
            moneda = str(row['MONEDA']).upper()
            
            if 'MILES' in moneda: # Miles de pesos
                total_clp += monto * 1000
            elif 'EURO' in moneda: # Euros
                total_clp += monto * EURO_TO_CLP_RATE
            elif 'DOLAR' in moneda: # Dolar
                total_clp += monto * DOLAR_TO_CLP_RATE
            else: # Asumimos Pesos normales
                total_clp += monto
        except:
            continue
            
    return total_clp

# --- FUNCIONES DE MATCHING RIGUROSO MEJORADAS ---

def extraer_tokens_apellidos(nombre_completo):
    """Extrae los tokens que probablemente son apellidos (parte final del string)."""
    partes = normalize_text(nombre_completo).split()
    if len(partes) <= 1: return set(partes)
    if len(partes) == 2: return {partes[1]} 
    return set(partes[-2:]) 

def match_nombre_riguroso(nombre_buscado, nombre_candidato):
    nombre_norm_buscado = normalize_text(nombre_buscado)
    nombre_norm_candidato = normalize_text(nombre_candidato)
    
    # 1. Match Exacto
    if nombre_norm_buscado == nombre_norm_candidato: return True
    
    # 2. Tokenización completa
    tokens_buscado = set(nombre_norm_buscado.split())
    tokens_candidato = set(nombre_norm_candidato.split())
    
    interseccion = tokens_buscado.intersection(tokens_candidato)
    
    if len(tokens_buscado) > 1 and len(interseccion) < 2:
        return False
        
    # 3. VERIFICACIÓN DE APELLIDOS
    apellidos_buscados = extraer_tokens_apellidos(nombre_buscado)
    match_apellido = False
    for ap in apellidos_buscados:
        if ap in tokens_candidato:
            match_apellido = True
            break
            
    if not match_apellido:
        return False

    return True

def verify_identity_strict(openalex_name, input_first, input_last):
    full_input = f"{input_first} {input_last}"
    return match_nombre_riguroso(full_input, openalex_name)

# --- ANID HISTORIC SEARCH ---
def search_anid_historic(first_name, last_name, df_historic):
    if df_historic is None or df_historic.empty: return pd.DataFrame(), ""
    
    full_input_name = f"{first_name} {last_name}"
    
    col_responsible = 'NOMBRE_RESPONSABLE'
    if col_responsible not in df_historic.columns:
        cols = [c for c in df_historic.columns if 'RESPONSABLE' in str(c).upper()]
        if cols: col_responsible = cols[0]
        else: return pd.DataFrame(), ""

    matches = df_historic[df_historic[col_responsible].apply(lambda x: match_nombre_riguroso(full_input_name, str(x)))].copy()
    
    anid_text_corpus = ""
    if not matches.empty:
        for _, row in matches.iterrows():
            title = str(row.get('NOMBRE_PROYECTO', ''))
            anid_text_corpus += (title + " ") * ANID_WEIGHT_MULTIPLIER
            
    return matches, anid_text_corpus

# --- EXTERNAL SEARCH (OPENALEX + ORCID + ORCID CHECK LOCAL) ---
def get_hybrid_research_text(first_name, last_name, df_orcid_local=None):
    oa_text = ""
    orcid_path = None
    display_name = f"{first_name} {last_name}"
    
    # 1. Búsqueda en OpenAlex
    try:
        query = f"{first_name} {last_name}"
        authors = Authors().search(query).get()
        target_author = None
        
        for auth in authors[:5]:
            if verify_identity_strict(auth['display_name'], first_name, last_name):
                target_author = auth
                display_name = auth['display_name']
                break
        
        if target_author:
            orcid_url = target_author.get('ids', {}).get('orcid')
            if orcid_url: orcid_path = orcid_url.replace("https://orcid.org/", "")
            
            if orcid_path and df_orcid_local is not None and not df_orcid_local.empty:
                match_local = df_orcid_local[df_orcid_local['ORCiD'].astype(str).str.contains(orcid_path, na=False)]
                if not match_local.empty:
                    col_nombre_local = next((c for c in df_orcid_local.columns if 'NOMBRE' in c.upper()), None)
                    if col_nombre_local:
                        display_name = match_local.iloc[0][col_nombre_local]

            works = Works().filter(author={"id": target_author['id']}).get()
            for w in works:
                t = w.get('title', '') or ''
                a = reconstruct_abstract(w.get('abstract_inverted_index'))
                k = " ".join([c['display_name'] for c in w.get('concepts', [])])
                oa_text += f"{t} {t} {k} {a} "
    except: pass

    # 2. Búsqueda directa en ORCID API
    orcid_text = ""
    if orcid_path:
        try:
            headers = {'Accept': 'application/json'}
            r = requests.get(f"https://pub.orcid.org/v3.0/{orcid_path}/works", headers=headers, timeout=5)
            if r.status_code == 200:
                groups = r.json().get('group', [])
                for g in groups:
                    s = g.get('work-summary', [])
                    if s:
                        t = s[0].get('title', {}).get('title', {}).get('value', '')
                        orcid_text += f"{t} {t} "
        except: pass
        
    return (oa_text + " " + orcid_text).strip(), display_name, orcid_path

# --- LOAD LOCAL DATABASES ---
@st.cache_data
def load_local_database():
    try:
        try: df_papers = pd.read_excel(FILE_PAPERS_VIGENTES)
        except: df_papers = pd.DataFrame()

        try:
            df_emails = pd.read_excel(FILE_EMAILS_ADMIN)
            if 'NOMBRES' in df_emails.columns and 'PRIMER_APELLIDO' in df_emails.columns:
                df_emails['Full_Name_Norm'] = (df_emails['NOMBRES'].astype(str) + " " + df_emails['PRIMER_APELLIDO'].astype(str) + " " + df_emails.get('SEGUNDO_APELLIDO', '').astype(str)).apply(normalize_text)
            else:
                name_col = [c for c in df_emails.columns if 'NOMBRE' in c.upper()][0]
                df_emails['Full_Name_Norm'] = df_emails[name_col].astype(str).apply(normalize_text)
        except: df_emails = pd.DataFrame()

        try: df_historic = pd.read_excel(FILE_ANID_HISTORIC)
        except: df_historic = pd.DataFrame()

        try: df_orcid_local = pd.read_excel(FILE_ORCID_USM)
        except: df_orcid_local = pd.DataFrame()

        # --- DATA PREP ---
        column_mapping = {'Keywords': 'Paper_Keywords', 'Title': 'Paper_Title', 'Abstract': 'Paper_Abstract', 'DOI': 'Paper_DOI'}
        df_papers.rename(columns=column_mapping, inplace=True)
        
        if 'Publication_Year' in df_papers.columns:
            if 'Paper_Year' in df_papers.columns: df_papers.drop(columns=['Paper_Year'], inplace=True)
            if 'Year' in df_papers.columns: df_papers.drop(columns=['Year'], inplace=True)
        else:
            if 'Paper_Year' in df_papers.columns: df_papers.rename(columns={'Paper_Year': 'Publication_Year'}, inplace=True)
            elif 'Year' in df_papers.columns: df_papers.rename(columns={'Year': 'Publication_Year'}, inplace=True)
        
        df_papers = df_papers.loc[:, ~df_papers.columns.duplicated()]
        if 'Publication_Year' not in df_papers.columns: df_papers['Publication_Year'] = 0
        df_papers['Publication_Year'] = pd.to_numeric(df_papers['Publication_Year'], errors='coerce').fillna(0).astype(int)

        def weighted_content(row):
            title = str(row.get('Paper_Title', ''))
            keywords = str(row.get('Paper_Keywords', ''))
            abstract = str(row.get('Paper_Abstract', ''))
            gold_keys = str(row.get('Gold_Keywords', ''))
            is_anid = str(row.get('Source')) == 'ANID_Project' or 'ANID' in str(row.get('Paper_DOI', ''))
            weighted_title = (title + " ") * (5 if is_anid else 3)
            base_text = f"{(gold_keys + ' ') * 8} {(keywords + ' ') * 6} {weighted_title} {(abstract + ' ') * 2}"
            if is_anid: return (base_text + " ") * ANID_WEIGHT_MULTIPLIER
            return base_text

        df_papers['Full_Content'] = df_papers.apply(weighted_content, axis=1)
        df_corpus = df_papers.groupby('Nombre_Busqueda')['Full_Content'].apply(' '.join).reset_index()
        
        return df_corpus, df_papers, df_historic, df_emails, df_orcid_local
        
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None, None
    
# --- UI SETUP ---
st.set_page_config(page_title="USM Recommender", layout="wide")

st.markdown("""
    <style>
    .stButton>button { background-color: #d93025; color: white; font-weight: bold; }
    .header-container { display: flex; align-items: center; }
    .logo-img { width: 60px; margin-left: 15px; }
    .anid-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #d93025; }
    .internal-alert { background-color: #e6fffa; padding: 15px; border-radius: 10px; border: 1px solid #38b2ac; margin-bottom: 20px;}
    h1 { color: #004b85; }
    </style>
    """, unsafe_allow_html=True)

col_header1, col_header2 = st.columns([0.8, 0.2])
with col_header1:
    st.markdown("<h1>Recommender USM Research <img src='https://upload.wikimedia.org/wikipedia/commons/4/47/Logo_UTFSM.png' class='logo-img'></h1>", unsafe_allow_html=True)
    st.subheader("Hybrid Authors Search (OpenAlex + ORCID) & ANID")

c1, c2 = st.columns(2)
with c1: first_name = st.text_input("First Name")
with c2: last_name = st.text_input("Last Name")

if 'search_performed' not in st.session_state: st.session_state.search_performed = False

if st.button("Analyze Profile & Find Collaborators"):
    st.session_state.search_performed = True
    st.session_state.first_name = first_name
    st.session_state.last_name = last_name

if st.session_state.search_performed:
    f_name = st.session_state.first_name
    l_name = st.session_state.last_name

    if f_name and l_name:
        with st.spinner('Loading databases and analyzing...'):
            df_corpus, df_raw_papers, df_historic_full, df_emails, df_orcid_local = load_local_database()
            
            if df_corpus is not None:
                # 1. Input Analysis
                user_text_ext, user_display_name, user_orcid = get_hybrid_research_text(f_name, l_name, df_orcid_local)
                
                name_parts = user_display_name.split()
                if len(name_parts) >= 2:
                    anid_query_first = name_parts[0]
                    anid_query_last = " ".join(name_parts[1:])
                else:
                    anid_query_first = f_name
                    anid_query_last = l_name

                user_anid_projects, user_text_anid = search_anid_historic(anid_query_first, anid_query_last, df_historic_full)
                final_user_text = (user_text_ext if user_text_ext else "") + " " + user_text_anid
                
                if not final_user_text.strip():
                    st.error("No sufficient research data found for this name.")
                else:
                    st.success(f"Profile Analyzed: **{user_display_name}**")
                    if user_orcid: st.caption(f"ORCID Detected: {user_orcid}")
                    
                    # --- ANID PORTFOLIO (HISTORIC) - DISPLAY DEL AUTOR BUSCADO ---
                    if not user_anid_projects.empty:
                        st.markdown("### ANID Project Portfolio (Historic)")
                        
                        # CALCULO DE TOTAL (CLP) PARA EL AUTOR DE ENTRADA
                        total_input_clp = calculate_total_clp(user_anid_projects)
                        st.markdown(f'<div class="anid-box">This researcher has registered ANID projects.<br><b>Total Adjudicated Funds:</b> ${total_input_clp:,.0f} (CLP)</div>', unsafe_allow_html=True)
                        
                        cols_show = ['NOMBRE_PROYECTO', 'AGNO_FALLO', 'DISCIPLINA_DETALLE','INSTITUCION_PRINCIPAL', 'MONTO_ADJUDICADO', 'MONEDA']
                        cols_exist = [c for c in cols_show if c in user_anid_projects.columns]
                        
                        # Normalizar Moneda para la vista (MODIFICADO A "Miles de pesos")
                        df_view = user_anid_projects[cols_exist].copy()
                        if 'MONEDA' in df_view.columns:
                            df_view['MONEDA'] = df_view['MONEDA'].astype(str).str.upper().apply(
                                lambda x: "Miles de pesos" if "MILES" in x else x
                            )

                        st.dataframe(
                            df_view.style.format({
                                "AGNO_FALLO": "{:.0f}", 
                                "MONTO_ADJUDICADO": lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else ""
                            }), 
                            hide_index=True
                        )
                    
                    # 2. Recommendation Algorithm
                    all_texts = [final_user_text] + df_corpus['Full_Content'].tolist()
                    professor_names = df_corpus['Nombre_Busqueda'].tolist()
                    
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                    
                    sorted_indices = cosine_sim.argsort()[::-1]
                    
                    # --- INTERNAL USER DETECTION LOGIC (RESTAURADA) ---
                    is_internal_user = False
                    self_match_score = 0.0
                    self_match_name = ""
                    
                    top_idx = sorted_indices[0]
                    top_score = cosine_sim[top_idx]
                    top_name = professor_names[top_idx]
                    
                    input_clean = normalize_text(user_display_name)
                    top_clean = normalize_text(top_name)
                    
                    # SI SE DETECTA QUE EL USUARIO ES EL MISMO PROFESOR:
                    if top_score > 0.90 or (match_nombre_riguroso(input_clean, top_clean) and top_score > 0.8):
                        is_internal_user = True
                        self_match_score = top_score
                        self_match_name = top_name
                        # AQUÍ SE QUITA EL INDICE 0 (EL PROPIO USUARIO) Y SE TOMAN LOS SIGUIENTES 5
                        final_top_indices = sorted_indices[1:6]
                    else:
                        final_top_indices = sorted_indices[:5]

                    # MOSTRAR ALERTA DE USUARIO INTERNO
                    if is_internal_user:
                        st.markdown(f"""
                        <div class="internal-alert">
                            <h4>ℹ️ Faculty Member Detected</h4>
                            <p>It appears you are already part of the USM faculty (<b>{self_match_name}</b>).</p>
                            <p>To provide useful recommendations, we have excluded your own profile from the suggestions list.</p>
                            <hr>
                            <b>System Self-Validation Score:</b> {self_match_score:.2%} (This confirms the algorithm correctly identified your research profile).
                        </div>
                        """, unsafe_allow_html=True)

                    results = []
                    for idx in final_top_indices:
                        p_name = professor_names[idx]
                        score = cosine_sim[idx]
                        
                        email = "Email not found"
                        dept = "Department not found"
                        if not df_emails.empty:
                            p_norm = normalize_text(p_name)
                            matches = df_emails[df_emails['Full_Name_Norm'].apply(lambda x: p_norm in x or x in p_norm)]
                            if not matches.empty:
                                email = matches.iloc[0].get('CORREO_INSTITUCIONAL', 'N/A')
                                dept = matches.iloc[0].get('DEPARTAMENTO', 'N/A')

                        # Info Fondos ANID
                        prof_projects, _ = search_anid_historic(p_name, "", df_historic_full)
                        funds_str = "0"
                        
                        # --- CALCULO UNIFICADO EN CLP ---
                        if not prof_projects.empty:
                            total_clp_collab = calculate_total_clp(prof_projects)
                            if total_clp_collab > 0:
                                funds_str = f"${total_clp_collab:,.0f} (CLP)"
                        
                        results.append({
                            'Professor': p_name, 
                            'Match Score': score, 
                            'Department': dept,
                            'Total Funds': funds_str,
                            'Email': email,
                            'Live_Projects': prof_projects
                        })

                    # --- DISPLAY RESULTS ---
                    st.markdown("### Collaboration Recommendations (USM)")
                    if is_internal_user:
                        st.write("These are your top potential collaborators (excluding yourself):")
                    else:
                        st.write("Top matches based on research topic similarity.")
                    
                    df_results = pd.DataFrame(results)
                    
                    st.dataframe(
                        df_results[["Professor", "Match Score", "Department", "Total Funds", "Email"]],
                        column_config={
                            "Professor": st.column_config.TextColumn("Professor Name", width="medium"),
                            "Match Score": st.column_config.ProgressColumn("Match Score", format="%.2f%%", min_value=0, max_value=1),
                            "Total Funds": st.column_config.TextColumn("Total Funds (CLP)", width="medium"),
                            "Email": st.column_config.LinkColumn("Email"),
                        },
                        hide_index=True,
                        width="stretch"
                    )

                    # Detail View
                    st.markdown("### Academic Detail & ANID Projects")
                    for res in results:
                        with st.expander(f"{res['Professor']}"):
                            c_info, c_papers = st.columns([1, 2])
                            with c_info:
                                st.write(f"**Department:** {res['Department']}")
                                st.write(f"**Email:** {res['Email']}")
                                st.write(f"**Funds:** {res['Total Funds']}")
                            
                            with c_papers:
                                df_anid_live = res['Live_Projects']
                                if not df_anid_live.empty:
                                    st.markdown("##### ANID Projects")
                                    col_year = 'AGNO_FALLO' if 'AGNO_FALLO' in df_anid_live.columns else 'AGNO_FALLO'
                                    
                                    if col_year in df_anid_live.columns:
                                        df_anid_live = df_anid_live.sort_values(col_year, ascending=False)
                                    
                                    for _, row in df_anid_live.iterrows():
                                        year_val = row.get(col_year, 0)
                                        year_str = str(int(float(year_val))) if pd.notna(year_val) and year_val != 0 else "----"
                                        title_val = row.get('NOMBRE_PROYECTO', 'Proyecto ANID')
                                        st.markdown(f"🔹 **[{year_str}]** {title_val}")
                                else: 
                                    st.write("No ANID projects found.")
                                
                                prof_data = df_raw_papers[df_raw_papers['Nombre_Busqueda'] == res['Professor']]
                                mask_anid = (prof_data['Source'] == 'ANID_Project') | (prof_data['Paper_DOI'].astype(str).str.contains('ANID', case=False, na=False))
                                df_only_papers = prof_data[~mask_anid].sort_values('Publication_Year', ascending=False).head(3)
                                
                                if not df_only_papers.empty:
                                    st.markdown("##### Recent Publications")
                                    for _, row in df_only_papers.iterrows():
                                        year_str = str(int(row.get('Publication_Year', 0)))
                                        st.markdown(f"- {year_str} | {row.get('Paper_Title','')}")