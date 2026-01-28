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

# --- FILE NAMES ---
FILE_PAPERS_VIGENTES = "Papers_Vigentes_Reparado_20260128_104709.xlsx" 
FILE_EMAILS_ADMIN = "ACAD-DOC 19_ENE_26.xlsx"
FILE_ANID_HISTORIC = "BDH_HISTORICA.xlsx"

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

# --- FUNCIONES DE MATCHING RIGUROSO---

def extraer_componentes_nombre(nombre_completo):
    partes = nombre_completo.strip().split()
    if len(partes) == 0: return ("", "", "", "")
    elif len(partes) == 1: return (partes[0], "", "", partes[0])
    elif len(partes) == 2: return (partes[0], partes[1], "", " ".join(partes))
    elif len(partes) == 3: return (partes[0], partes[1], partes[2], " ".join(partes))
    else:
        nombres = " ".join(partes[:-2])
        return (nombres, partes[-2], partes[-1], " ".join(partes))

#Compara el INPUT del usuario con el NOMBRE DE LA BASE DE DATOS.
#Retorna True si es la misma persona.

def match_nombre_riguroso(nombre_buscado, nombre_candidato):

    nombre_norm_buscado = normalize_text(nombre_buscado)
    nombre_norm_candidato = normalize_text(nombre_candidato)
    
    # 1. Match exacto
    if nombre_norm_buscado == nombre_norm_candidato: return True
    
    # 2. Análisis de componentes
    nombres_b, ap_pat_b, ap_mat_b, _ = extraer_componentes_nombre(nombre_norm_buscado)
    nombres_c, ap_pat_c, ap_mat_c, _ = extraer_componentes_nombre(nombre_norm_candidato)
    
    # Regla: Apellido Paterno debe coincidir (o estar contenido si uno es compuesto)
    if not ap_pat_b or not ap_pat_c: return False
    if ap_pat_b != ap_pat_c:
        # Check cruzado para apellidos compuestos cortados
        if ap_pat_b not in ap_pat_c and ap_pat_c not in ap_pat_b:
            return False

    # Regla: Primer nombre debe coincidir
    prim_nom_b = nombres_b.split()[0] if nombres_b else ""
    prim_nom_c = nombres_c.split()[0] if nombres_c else ""
    if prim_nom_b != prim_nom_c:
        # Intento fuzzy leve para typos en nombre (Andres vs Andre)
        if SequenceMatcher(None, prim_nom_b, prim_nom_c).ratio() < 0.85:
            return False

    # Regla: Si ambos tienen materno, debe coincidir
    if ap_mat_b and ap_mat_c:
        if ap_mat_b != ap_mat_c: return False
        
    return True

def verify_identity_strict(openalex_name, input_first, input_last):
    # Usamos la nueva lógica robusta también para OpenAlex
    full_input = f"{input_first} {input_last}"
    return match_nombre_riguroso(full_input, openalex_name)

# --- ANID HISTORIC SEARCH (ACTUALIZADA) ---
def search_anid_historic(first_name, last_name, df_historic):
    if df_historic is None or df_historic.empty: return pd.DataFrame(), ""
    
    # Construir nombre completo del input
    full_input_name = f"{first_name} {last_name}"
    
    col_responsible = 'NOMBRE_RESPONSABLE'
    if col_responsible not in df_historic.columns:
        cols = [c for c in df_historic.columns if 'RESPONSABLE' in str(c).upper()]
        if cols: col_responsible = cols[0]
        else: return pd.DataFrame(), ""

    # Usamos apply con la función robusta
    # Nota: str(x) maneja posibles NaNs en la base de datos
    matches = df_historic[df_historic[col_responsible].apply(lambda x: match_nombre_riguroso(full_input_name, str(x)))].copy()
    
    anid_text_corpus = ""
    if not matches.empty:
        for _, row in matches.iterrows():
            title = str(row.get('NOMBRE_PROYECTO', ''))
            anid_text_corpus += (title + " ") * ANID_WEIGHT_MULTIPLIER
            
    return matches, anid_text_corpus

# --- EL RESTO DEL CÓDIGO (get_hybrid_research_text, load_local_database, UI) SIGUE IGUAL ---
# Solo asegúrate de copiar el resto del archivo app_anid_usm.py debajo de esto.


# --- EXTERNAL SEARCH (OPENALEX + ORCID) ---
def get_hybrid_research_text(first_name, last_name):
    oa_text = ""
    orcid_path = None
    display_name = f"{first_name} {last_name}"
    
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
            
            works = Works().filter(author={"id": target_author['id']}).get()
            for w in works:
                t = w.get('title', '') or ''
                a = reconstruct_abstract(w.get('abstract_inverted_index'))
                k = " ".join([c['display_name'] for c in w.get('concepts', [])])
                oa_text += f"{t} {t} {k} {a} "
    except: pass

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
        
    return (oa_text + " " + orcid_text).strip(), display_name

# --- LOAD LOCAL DATABASES (CORREGIDA - FIX DUPLICADOS) ---
@st.cache_data
def load_local_database():
    try:
        # 1. Load Papers
        try:
            df_papers = pd.read_excel(FILE_PAPERS_VIGENTES)
        except FileNotFoundError:
            st.error(f"File not found: {FILE_PAPERS_VIGENTES}.")
            return None, None, None, None

        # 2. Load Emails (ACAD-DOC)
        try:
            df_emails = pd.read_excel(FILE_EMAILS_ADMIN)
            # Create normalized column for matching
            if 'NOMBRES' in df_emails.columns and 'PRIMER_APELLIDO' in df_emails.columns:
                df_emails['Full_Name_Norm'] = (
                    df_emails['NOMBRES'].astype(str) + " " + 
                    df_emails['PRIMER_APELLIDO'].astype(str) + " " + 
                    df_emails.get('SEGUNDO_APELLIDO', '').astype(str)
                ).apply(normalize_text)
            else:
                name_col = [c for c in df_emails.columns if 'NOMBRE' in c.upper()][0]
                df_emails['Full_Name_Norm'] = df_emails[name_col].astype(str).apply(normalize_text)

        except Exception as e:
            st.warning(f"Could not load email file: {e}")
            df_emails = pd.DataFrame()

        # 3. Load ANID Historic
        try:
            df_historic = pd.read_excel(FILE_ANID_HISTORIC)
        except:
            df_historic = pd.DataFrame()

        # --- FIX ROBUSTO DE COLUMNAS Y DUPLICADOS ---
        # 1. Renombrar columnas seguras (que no son Año)
        column_mapping = {
            'Keywords': 'Paper_Keywords', 
            'Title': 'Paper_Title',
            'Abstract': 'Paper_Abstract', 
            'DOI': 'Paper_DOI'
            # Quitamos 'Year' y 'Paper_Year' de aquí para manejarlo con lógica condicional
        }
        df_papers.rename(columns=column_mapping, inplace=True)
        
        # 2. Manejo inteligente del Año para evitar colisiones
        if 'Publication_Year' in df_papers.columns:
            # Si ya existe (del generador nuevo), eliminamos las versiones antiguas para que no molesten
            if 'Paper_Year' in df_papers.columns:
                df_papers.drop(columns=['Paper_Year'], inplace=True)
            if 'Year' in df_papers.columns:
                df_papers.drop(columns=['Year'], inplace=True)
        else:
            # Si no existe, buscamos cual renombrar
            if 'Paper_Year' in df_papers.columns:
                df_papers.rename(columns={'Paper_Year': 'Publication_Year'}, inplace=True)
            elif 'Year' in df_papers.columns:
                df_papers.rename(columns={'Year': 'Publication_Year'}, inplace=True)
        
        # 3. Limpieza final de duplicados (por si acaso quedaron otras columnas repetidas)
        df_papers = df_papers.loc[:, ~df_papers.columns.duplicated()]

        # 4. Asegurar formato numérico
        if 'Publication_Year' not in df_papers.columns:
            df_papers['Publication_Year'] = 0
            
        df_papers['Publication_Year'] = pd.to_numeric(df_papers['Publication_Year'], errors='coerce').fillna(0).astype(int)

        # Apply Weighting
        def weighted_content(row):
            title = str(row.get('Paper_Title', ''))
            keywords = str(row.get('Paper_Keywords', ''))
            abstract = str(row.get('Paper_Abstract', ''))
            gold_keys = str(row.get('Gold_Keywords', ''))
            if gold_keys == 'nan': gold_keys = ""

            is_anid = str(row.get('Source')) == 'ANID_Project' or 'ANID' in str(row.get('Paper_DOI', ''))

            weighted_title = (title + " ") * (5 if is_anid else 3)
            weighted_keywords = (keywords + " ") * 6
            weighted_gold = (gold_keys + " ") * 8
            weighted_abstract = (abstract + " ") * 2
            
            base_text = f"{weighted_gold} {weighted_gold} {weighted_keywords} {weighted_keywords} {weighted_title} {weighted_title} {weighted_abstract}"
            if is_anid: return (base_text + " ") * ANID_WEIGHT_MULTIPLIER
            return base_text

        df_papers['Full_Content'] = df_papers.apply(weighted_content, axis=1)
        df_corpus = df_papers.groupby('Nombre_Busqueda')['Full_Content'].apply(' '.join).reset_index()
        
        return df_corpus, df_papers, df_historic, df_emails
        
    except Exception as e:
        st.error(f"Critical error loading files: {e}")
        return None, None, None, None
    
# --- UI SETUP ---
st.set_page_config(page_title="USM Recommender", layout="wide")

# CSS Styling for Logo and Layout
st.markdown("""
    <style>
    .stButton>button {
        background-color: #d93025; 
        color: white; 
        font-weight: bold;
    }
    .header-container { 
        display: flex; 
        align-items: center; 
    }
    .logo-img { 
        width: 60px; 
        margin-left: 15px; 
    }
    .anid-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #d93025;
    }
    h1 { color: #004b85; }
    </style>
    """, unsafe_allow_html=True)

# Header with Logo
col_header1, col_header2 = st.columns([0.8, 0.2])
with col_header1:
    st.markdown("""
        <div class='header-container'>
            <h1>Recommender USM Research (ANID) <img src='https://upload.wikimedia.org/wikipedia/commons/4/47/Logo_UTFSM.png' class='logo-img'></h1>
        </div>
        """, unsafe_allow_html=True)
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
            df_corpus, df_raw_papers, df_historic_full, df_emails = load_local_database()
            
            if df_corpus is not None:
                # 1. Input Analysis
                user_text_ext, user_display_name = get_hybrid_research_text(f_name, l_name)
                user_anid_projects, user_text_anid = search_anid_historic(f_name, l_name, df_historic_full)
                final_user_text = (user_text_ext if user_text_ext else "") + " " + user_text_anid
                
                if not final_user_text.strip():
                    st.error("No sufficient research data found for this name.")
                else:
                    st.success(f"Profile Analyzed: **{user_display_name}**")
                    
                    # --- SECTION: ANID PROJECT PORTFOLIO (HISTORIC) ---
                    if not user_anid_projects.empty:
                        st.markdown("### ANID Project Portfolio (Historic)")
                        st.markdown('<div class="anid-box">This researcher has registered ANID projects.</div>', unsafe_allow_html=True)
                        
                        cols_show = ['NOMBRE_PROYECTO', 'AGNO_CONCURSO', 'DISCIPLINA_DETALLE', 'INSTITUCION_PRINCIPAL', 'MONTO_ADJUDICADO', 'MONEDA']
                        cols_exist = [c for c in cols_show if c in user_anid_projects.columns]
                        
                        # CORRECCIÓN 1: Formato en la Tabla
                        # Usamos una función lambda para formatear el número con comas y luego reemplazarlas por puntos
                        st.dataframe(
                            user_anid_projects[cols_exist].style.format({
                                "AGNO_CONCURSO": "{:.0f}", 
                                "MONTO_ADJUDICADO": lambda x: f"{x:,.0f}".replace(",", ".") if pd.notna(x) else ""
                            }), 
                            hide_index=True
                        )
                        
                        # Funds Calculation
                        if 'MONTO_ADJUDICADO' in user_anid_projects.columns and 'MONEDA' in user_anid_projects.columns:
                            st.markdown("#### Total Adjudicated Funds")
                            totals = user_anid_projects.groupby('MONEDA')['MONTO_ADJUDICADO'].sum()
                            cols_metric = st.columns(len(totals))
                            for i, (currency, amount) in enumerate(totals.items()):
                                with cols_metric[i]:
                                    # CORRECCIÓN 2: Formato en los Indicadores (Metrics)
                                    # Formateamos con coma (estándar US) y luego reemplazamos la coma por punto
                                    formatted_value = f"{amount:,.0f}".replace(",", ".")
                                    st.metric(label=f"Total in {currency}", value=formatted_value)
                    
                    # 2. Recommendation Algorithm (TF-IDF)
                    all_texts = [final_user_text] + df_corpus['Full_Content'].tolist()
                    professor_names = df_corpus['Nombre_Busqueda'].tolist()
                    
                    vectorizer = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                        
                    top_indices = cosine_sim.argsort()[-5:][::-1]
                    
                    results = []
                    for idx in top_indices:
                        p_name = professor_names[idx]
                        score = cosine_sim[idx]
                        
                        # --- EMAIL & DEPT LOOKUP ---
                        email = "Email not found"
                        dept = "Department not found"
                        
                        if not df_emails.empty:
                            p_norm = normalize_text(p_name)
                            matches = df_emails[df_emails['Full_Name_Norm'].apply(lambda x: p_norm in x or x in p_norm)]
                            if not matches.empty:
                                email = matches.iloc[0].get('CORREO_INSTITUCIONAL', 'N/A')
                                dept = matches.iloc[0].get('DEPARTAMENTO', 'N/A')

                        # --- FUNDS CALCULATION (NUEVO) ---
                        # Buscamos los proyectos históricos de este profesor recomendado
                        # Usamos search_anid_historic pasando p_name como first_name y vacío como last_name
                        # --- FUNDS CALCULATION (CORREGIDO) ---
                        # --- FUNDS CALCULATION (CORREGIDO & LIMPIO) ---
                        prof_projects, _ = search_anid_historic(p_name, "", df_historic_full)
                        
                        funds_str = "0"
                        if not prof_projects.empty and 'MONTO_ADJUDICADO' in prof_projects.columns and 'MONEDA' in prof_projects.columns:
                            # 1. NORMALIZACIÓN
                            prof_projects = prof_projects.copy()
                            prof_projects['MONEDA_NORM'] = prof_projects['MONEDA'].astype(str).str.strip().str.title()
                            
                            # 2. AGRUPAR
                            totals = prof_projects.groupby('MONEDA_NORM')['MONTO_ADJUDICADO'].sum()
                            
                            parts = []
                            for currency, amount in totals.items():
                                # FIX: Omitir si la moneda es 'Nan' o vacía
                                if str(currency).lower() == 'nan' or str(currency).strip() == '': continue
                                
                                fmt_amount = f"{amount:,.0f}".replace(",", ".")
                                parts.append(f"{currency}: ${fmt_amount}")
                            
                            if parts:
                                funds_str = " | ".join(parts)
                        
                        results.append({
                            'Professor': p_name, 
                            'Match Score': score, 
                            'Department': dept,
                            'Total Funds': funds_str,
                            'Email': email,
                            'Live_Projects': prof_projects  # <--- GUARDAMOS LOS PROYECTOS AQUÍ
                        })

                    # --- SECTION: COLLABORATION RECOMMENDATIONS ---
                    st.markdown("### Collaboration Recommendations (USM)")
                    st.write(f"Matches based on topic similarity, with **high priority** on ANID Project overlaps.")
                    
                    df_results = pd.DataFrame(results)
                    
                    # COLUMNAS A MOSTRAR (Filtramos para que no salga Live_Projects)
                    cols_visible = ["Professor", "Match Score", "Department", "Total Funds", "Email"]
                    
                    st.dataframe(
                        df_results[cols_visible],
                        column_config={
                            "Professor": st.column_config.TextColumn("Professor Name", width="medium"),
                            "Match Score": st.column_config.ProgressColumn(
                                "Match Score", 
                                format="%.2f%%", 
                                min_value=0, 
                                max_value=1
                            ),
                            "Department": st.column_config.TextColumn("Department", width="small"),
                            "Total Funds": st.column_config.TextColumn("Total Adjudicated Funds", width="medium"), # <--- Configuración Visual
                            "Email": st.column_config.LinkColumn("Email"),
                        },
                        hide_index=True,
                        width="stretch"
                    )

                    # 3. Detailed View
                    # 3. Detailed View
                    # 3. Detailed View
                    st.markdown("### Academic Detail & ANID Projects")
                    for res in results:
                        p_name = res['Professor']
                        with st.expander(f"{p_name}"):
                            c_info, c_papers = st.columns([1, 2])
                            with c_info:
                                st.write(f"**Department:** {res['Department']}")
                                st.write(f"**Email:** {res['Email']}")
                                st.write(f"**Funds:** {res['Total Funds']}") # Dato extra útil
                            
                            with c_papers:
                                # 1. PROYECTOS ANID (Fuente: Búsqueda en Vivo - La misma que calcula el dinero)
                                df_anid_live = res['Live_Projects']
                                
                                if not df_anid_live.empty:
                                    st.markdown("##### All ANID Projects (Verified)")
                                    
                                    # Intentar detectar columnas de Año y Título dinámicamente
                                    col_year = next((c for c in df_anid_live.columns if 'AGNO' in c or 'ANO' in c), None)
                                    col_title = next((c for c in df_anid_live.columns if 'NOMBRE' in c and 'PROYECTO' in c), 'NOMBRE_PROYECTO')
                                    
                                    # Ordenar si tenemos columna de año
                                    if col_year:
                                        df_anid_live = df_anid_live.sort_values(col_year, ascending=False)
                                    
                                    for _, row in df_anid_live.iterrows():
                                        # Obtener año limpio
                                        year_val = row.get(col_year, 0) if col_year else 0
                                        try: year_str = str(int(float(year_val))) if year_val != 0 else "----"
                                        except: year_str = "----"
                                        
                                        title_val = row.get(col_title, "Proyecto ANID")
                                        st.markdown(f"🔹 **[{year_str}]** {title_val}")
                                else: 
                                    st.write("No ANID projects found for this researcher.")
                                
                                # 2. PUBLICACIONES (Fuente: OpenAlex - Excel Estático)
                                prof_data = df_raw_papers[df_raw_papers['Nombre_Busqueda'] == p_name]
                                
                                # Filtramos para NO mostrar proyectos ANID duplicados aquí, solo Papers
                                mask_anid = (prof_data['Source'] == 'ANID_Project') | (prof_data['Paper_DOI'].astype(str).str.contains('ANID', case=False, na=False))
                                df_only_papers = prof_data[~mask_anid].sort_values('Publication_Year', ascending=False).head(3)
                                
                                if not df_only_papers.empty:
                                    st.markdown("##### Recent Publications (Top 3)")
                                    for _, row in df_only_papers.iterrows():
                                        year = row.get('Publication_Year', 0)
                                        year_str = str(int(year)) if pd.notna(year) and year != 0 else "----"
                                        st.markdown(f"- {year_str} | {row.get('Paper_Title','')}")