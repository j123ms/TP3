import streamlit as st
import pandas as pd
import os
import warnings
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score # <-- NUEVO: Importación para validación cruzada

warnings.filterwarnings('ignore')

# Configuración de la interfaz web
st.set_page_config(page_title="Evaluación Oncológica Integral", page_icon="⚕️", layout="wide")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #4A90E2;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #757575;
        border-top: 1px solid #333;
        padding-top: 10px;
        margin-top: 50px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def inicializar_modelo():
    ruta_zip = 'DSCancerGastrointestinal.zip' 
    
    if not os.path.exists(ruta_zip):
        return None, None, None, None, None, None # Se agregaron más Nones para los nuevos retornos
        
    dataFrame = pd.read_csv(ruta_zip, sep=';', compression='zip')
    
    columnas_a_borrar = ['Biopsia', 'Endoscopia', 'Tomografia']
    dataFrame = dataFrame.drop(columns=columnas_a_borrar, errors='ignore')
    dataFrame['Resultados'] = dataFrame['Resultados'].astype('int')
    
    if 'Condiciones' in dataFrame.columns:
        condiciones_cat = dataFrame['Condiciones'].astype('category')
        mapa_condiciones = {categoria: codigo for codigo, categoria in enumerate(condiciones_cat.cat.categories)}
        dataFrame['Condiciones'] = condiciones_cat.cat.codes
    else:
        mapa_condiciones = {"None": 0}

    df_numeric = dataFrame.select_dtypes(include=['number'])

    df_sanos = df_numeric[df_numeric['Resultados'] == 0]
    df_enfermos = df_numeric[df_numeric['Resultados'] == 1]

    codigo_none = mapa_condiciones.get("None", 0)
    filtro_puros = (
        (df_sanos['Fumador'] == 0) & 
        (df_sanos['Alcohol'] == 0) & 
        (df_sanos['HistoFamiliar'] == 0) & 
        (df_sanos['Dieta'] == 0) &
        (df_sanos['helicobacter_pylori_infection'] == 0) & 
        (df_sanos['Condiciones'] == codigo_none)
    )

    sanos_puros = df_sanos[filtro_puros]
    sanos_comunes = df_sanos[~filtro_puros]

    cantidad_necesaria = len(df_enfermos)
    cantidad_puros = min(len(sanos_puros), cantidad_necesaria // 2)
    cantidad_comunes = cantidad_necesaria - cantidad_puros

    muestra_puros = sanos_puros.sample(n=cantidad_puros, random_state=42)
    muestra_comunes = sanos_comunes.sample(n=cantidad_comunes, random_state=42)

    df_sanos_final = pd.concat([muestra_puros, muestra_comunes])
    df_calibrado = pd.concat([df_sanos_final, df_enfermos]).sample(frac=1, random_state=42)

    X = df_calibrado.drop(columns=['Resultados'])
    y = df_calibrado['Resultados']
    
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    # --- NUEVO: VALIDACIÓN CRUZADA (Cross-Validation) ---
    # Realizamos una validación cruzada con 5 particiones (folds)
    # Esto evalúa el modelo internamente antes de hacer el fit final
    scores = cross_val_score(et_model, X, y, cv=5, scoring='accuracy')
    cv_mean = scores.mean()
    cv_std = scores.std()
    # ----------------------------------------------------
    
    # Entrenamiento final con toda la data para usar en producción
    et_model.fit(X, y)
    
    columnas_modelo = X.columns.tolist()
    
    return et_model, columnas_modelo, df_numeric, mapa_condiciones, cv_mean, cv_std

# Cargar el modelo (ahora desempaqueta también las métricas de CV)
et_model, columnas_modelo, df_numeric, mapa_condiciones, cv_mean, cv_std = inicializar_modelo()

if et_model is None:
    st.error("Error crítico: No se encontró la base de datos 'DSCancerGastrointestinal.zip'.")
    st.stop()

# --- PANEL LATERAL: INGRESO DE DATOS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=60) # Ícono médico referencial
    st.markdown("### Perfil del Paciente")
    
    edad = st.number_input("Edad (18-100):", min_value=18, max_value=100, value=45, step=1)
    genero = st.selectbox("Género:", ["Masculino", "Femenino"])
    familia = st.selectbox("Antecedentes Familiares:", ["No", "Sí"])
    
    st.markdown("---")
    st.markdown("### Factores de Riesgo")
    fumador = st.toggle("Fumador")
    alcohol = st.toggle("Consumo de Alcohol")
    dieta = st.toggle("Dieta alta en procesados/grasas")
    h_pylori = st.toggle("Infección por H. Pylori")
    
    st.markdown("### Condición Preexistente")
    opciones_visuales = ["Ninguna", "Gastritis Crónica", "Diabetes"]
    condicion_seleccionada = st.selectbox("Seleccione:", opciones_visuales)
    
    st.markdown("<br>", unsafe_allow_html=True)
    ejecutar_analisis = st.button("Evaluar Riesgo Clínico", type="primary")

    # --- NUEVO: MOSTRAR RESULTADOS DE VALIDACIÓN CRUZADA ---
    st.markdown("---")
    st.markdown("### 📊 Rendimiento del Modelo")
    st.caption(f"**Precisión (CV 5-Folds):** {cv_mean * 100:.2f}%")
    st.caption(f"**Margen de error (Desviación):** ± {cv_std * 100:.2f}%")
    # --------------------------------------------------------

# --- PANEL PRINCIPAL: RESULTADOS ---
st.markdown('<p class="main-header">⚕️ Evaluación Oncológica Integral</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de apoyo a la decisión clínica.</p>', unsafe_allow_html=True)

if ejecutar_analisis:
    with st.spinner('Procesando variables clínicas y calculando probabilidad...'):
        genero_val = 1 if genero == "Masculino" else 0
        familia_val = 1 if familia == "Sí" else 0
        fumador_val = 1 if fumador else 0
        alcohol_val = 1 if alcohol else 0
        dieta_val = 1 if dieta else 0
        h_pylori_val = 1 if h_pylori else 0

        traductor_condiciones = {
            "Ninguna": "None",
            "Gastritis Crónica": "Chronic Gastritis",
            "Diabetes": "Diabetes"
        }
        
        val_original = traductor_condiciones[condicion_seleccionada]
        condicion_val = mapa_condiciones.get(val_original, 0)

        # Inyectar medianas para variables de fondo
        full_data = {col: [df_numeric[col].median()] for col in columnas_modelo}
        
        # Asignar inputs del usuario con estricta validación
        if 'Edad' in full_data: full_data['Edad'] = [int(edad)]
        if 'Genero' in full_data: full_data['Genero'] = [genero_val]
        if 'HistoFamiliar' in full_data: full_data['HistoFamiliar'] = [familia_val]
        if 'Fumador' in full_data: full_data['Fumador'] = [fumador_val]
        if 'Alcohol' in full_data: full_data['Alcohol'] = [alcohol_val]
        if 'Dieta' in full_data: full_data['Dieta'] = [dieta_val]
        if 'helicobacter_pylori_infection' in full_data: full_data['helicobacter_pylori_infection'] = [h_pylori_val]
        if 'Condiciones' in full_data: full_data['Condiciones'] = [condicion_val]
        
        df_input = pd.DataFrame(full_data)[columnas_modelo]

        # Predicción
        probs = et_model.predict_proba(df_input)[0]
        prob_riesgo = float(probs[1] * 100)
        prob_formateada = f"{prob_riesgo:.2f} %" # Formateo estricto a 2 decimales sin notación científica

        st.markdown("### Resumen de Evaluación")
        st.divider()

        # Presentación Formal de Resultados
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Probabilidad Predictiva", value=prob_formateada)

        with col_res2:
            if prob_riesgo >= 50:
                st.error("### 🔴 Nivel: ALTO RIESGO\n\n**Recomendación:** Se requiere una evaluación especializada.")
            elif 35 <= prob_riesgo < 50:
                st.warning("### 🟡 Nivel: RIESGO MODERADO\n\n**Recomendación:** Se sugiere programar una consulta médica preventiva.")
            else:
                st.success("### 🟢 Nivel: BAJO RIESGO\n\n**Recomendación:** Perfil clínico favorable de acuerdo a las variables analizadas. Mantener hábitos saludables y controles rutinarios.")

else:
    st.info("👈 Por favor, complete los datos en el panel lateral y haga clic en **'Evaluar Riesgo Clínico'** para generar el reporte.")

# Disclaimer Legal/Formal
st.markdown('<div class="disclaimer"><strong>Aviso legal:</strong> Esta herramienta utiliza modelos de aprendizaje automático basados en datos estadísticos para estimar factores de riesgo. Los resultados proporcionados son de carácter informativo y de apoyo a la decisión. Bajo ninguna circunstancia sustituyen un diagnóstico clínico profesional, consejo médico o evaluación presencial realizada por un especialista calificado.</div>', unsafe_allow_html=True)