import streamlit as st
import pandas as pd
import os
import warnings
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

# Configuración de la interfaz web
st.set_page_config(page_title="Screening Oncológico Integral", page_icon="⚕️", layout="centered")

@st.cache_resource
def inicializar_modelo():
    ruta_zip = 'DSCancerGastrointestinal.zip' 
    
    if not os.path.exists(ruta_zip):
        return None, None, None, None
        
    dataFrame = pd.read_csv(ruta_zip, sep=';', compression='zip')
    
    columnas_a_borrar = ['Biopsia', 'Endoscopia', 'Tomografia']
    dataFrame = dataFrame.drop(columns=columnas_a_borrar, errors='ignore')
    dataFrame['Resultados'] = dataFrame['Resultados'].astype('int')
    
    # Procesamiento de "Condiciones"
    if 'Condiciones' in dataFrame.columns:
        condiciones_cat = dataFrame['Condiciones'].astype('category')
        mapa_condiciones = {categoria: codigo for codigo, categoria in enumerate(condiciones_cat.cat.categories)}
        dataFrame['Condiciones'] = condiciones_cat.cat.codes
    else:
        mapa_condiciones = {"None": 0}

    df_numeric = dataFrame.select_dtypes(include=['number'])

    df_sanos = df_numeric[df_numeric['Resultados'] == 0]
    df_enfermos = df_numeric[df_numeric['Resultados'] == 1]

    # Filtro de calibración para que el modelo identifique correctamente el peso de las variables
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
    cantidad_puros = min(len(sanos_puros), quantity_necesaria // 2 if 'quantity_necesaria' in locals() else cantidad_necesaria // 2)
    cantidad_comunes = cantidad_necesaria - cantidad_puros

    muestra_puros = sanos_puros.sample(n=cantidad_puros, random_state=42)
    muestra_comunes = sanos_comunes.sample(n=cantidad_comunes, random_state=42)

    df_sanos_final = pd.concat([muestra_puros, muestra_comunes])
    df_calibrado = pd.concat([df_sanos_final, df_enfermos]).sample(frac=1, random_state=42)

    X = df_calibrado.drop(columns=['Resultados'])
    y = df_calibrado['Resultados']
    
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(X, y)
    
    columnas_modelo = X.columns.tolist()
    
    return et_model, columnas_modelo, df_numeric, mapa_condiciones

# Cargar el modelo
et_model, columnas_modelo, df_numeric, mapa_condiciones = inicializar_modelo()

# Interfaz de Usuario
st.title("⚕️ Screening Oncológico Integral")
st.markdown("### Ingrese los datos clínicos del paciente:")

if et_model is None:
    st.error("Error crítico: No se encontró la base de datos 'DSCancerGastrointestinal.zip'.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad del paciente (18-100):", min_value=18, max_value=100, value=45, step=1)
    familia = st.radio("Antecedentes Familiares:", ["No", "Sí"], horizontal=True)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    genero = st.radio("Género:", ["Masculino", "Femenino"], horizontal=True)

st.markdown("---")
st.markdown("#### Hábitos de Riesgo y Factores Clínicos:")

col3, col4 = st.columns(2)
with col3:
    fumador = st.checkbox("Fumador")
    alcohol = st.checkbox("Consumo de Alcohol")
    dieta = st.checkbox("Dieta alta en procesados/grasas")
with col4:
    h_pylori = st.checkbox("Infección por H. Pylori (Activa/Previa)")
    
    opciones_visuales = ["Ninguna", "Gastritis Crónica", "Diabetes"]
    condicion_seleccionada = st.selectbox("Condición Preexistente:", opciones_visuales)

st.markdown("---")

if st.button("Evaluar Riesgo Clínico", type="primary"):
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

    # Inyectar medianas para las variables genómicas de fondo
    full_data = {col: [df_numeric[col].median()] for col in columnas_modelo}
    
    # Asignar inputs del usuario
    if 'Edad' in full_data: full_data['Edad'] = [edad]
    if 'Genero' in full_data: full_data['Genero'] = [genero_val]
    if 'HistoFamiliar' in full_data: full_data['HistoFamiliar'] = [familia_val]
    if 'Fumador' in full_data: full_data['Fumador'] = [fumador_val]
    if 'Alcohol' in full_data: full_data['Alcohol'] = [alcohol_val]
    if 'Dieta' in full_data: full_data['Dieta'] = [dieta_val]
    if 'helicobacter_pylori_infection' in full_data: full_data['helicobacter_pylori_infection'] = [h_pylori_val]
    if 'Condiciones' in full_data: full_data['Condiciones'] = [condicion_val]
    
    df_input = pd.DataFrame(full_data)
    df_input = df_input[columnas_modelo]

    # Predicción y cálculo de probabilidad
    probs = et_model.predict_proba(df_input)[0]
    prob_riesgo = probs[1] * 100

    # --- LÓGICA DE SALIDA TIPO SEMÁFORO ---
    st.markdown("### Resultado de la Evaluación:")
    
    if prob_riesgo >= 65:
        st.error(f"**🔴 ALTO RIESGO**\n\nProbabilidad Predictiva: **{prob_riesgo:.1f}%**\n\nSe recomienda priorizar de manera urgente una endoscopia digestiva alta y evaluación especializada inmediata.")
    elif 35 <= prob_riesgo < 65:
        st.warning(f"**🟡 RIESGO MODERADO**\n\nProbabilidad Predictiva: **{prob_riesgo:.1f}%**\n\nSe sugiere programar una consulta de seguimiento clínico, control de factores de riesgo y monitoreo periódico de síntomas.")
    else:
        st.success(f"**🟢 BAJO RIESGO**\n\nProbabilidad Predictiva: **{prob_riesgo:.1f}%**\n\nPerfil clínico favorable según los factores analizados. Continuar con controles rutinarios y hábitos saludables.")