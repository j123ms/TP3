import streamlit as st
import pandas as pd
import os
import warnings
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')

# Configuración de la interfaz web
st.set_page_config(page_title="Triaje Clínico Preventivo", page_icon="⚕️", layout="centered")

# Utilizamos caché para que el modelo solo se entrene una vez al iniciar el servidor
@st.cache_resource
def inicializar_modelo():
    ruta_zip = 'DSCancerGastrointestinal.zip' # Ahora apuntamos al archivo comprimido
    
    if not os.path.exists(ruta_zip):
        return None, None, None
        
    # Agregamos compression='zip' para que Pandas lo lea directamente
    dataFrame = pd.read_csv(ruta_zip, sep=';', compression='zip')
    
    columnas_a_borrar = ['Biopsia', 'Endoscopia', 'Tomografia']
    dataFrame = dataFrame.drop(columns=columnas_a_borrar, errors='ignore')
    dataFrame['Resultados'] = dataFrame['Resultados'].astype('int')
    df_numeric = dataFrame.select_dtypes(include=['number']).fillna(0)

    df_sanos = df_numeric[df_numeric['Resultados'] == 0]
    df_enfermos = df_numeric[df_numeric['Resultados'] == 1]

    filtro_puros = (
        (df_sanos['Fumador'] == 0) & 
        (df_sanos['Alcohol'] == 0) & 
        (df_sanos['HistoFamiliar'] == 0) & 
        (df_sanos['Dieta'] == 0)
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
    et_model.fit(X, y)
    
    columnas_modelo = X.columns.tolist()
    
    return et_model, columnas_modelo, df_numeric

# Cargar el modelo
et_model, columnas_modelo, df_numeric = inicializar_modelo()

# Interfaz de Usuario
st.title("⚕️ Triaje Clínico Preventivo")
st.markdown("### Ingrese los datos solicitados del paciente:")

if et_model is None:
    st.error("Error crítico: No se encontró la base de datos 'DSCancerGastrointestinal.zip'.")
    st.stop()

# Formularios en columnas
col1, col2 = st.columns(2)

with col1:
    # La edad sigue usando number_input, lo que permite teclear o usar flechas (con validación estricta de 18 a 100)
    edad = st.number_input("Edad del paciente (18-100):", min_value=18, max_value=100, value=45, step=1)
    
    # Cambiamos selectbox por radio horizontal para forzar la selección por clic
    familia = st.radio("Antecedentes Familiares:", ["No", "Sí"], horizontal=True)

with col2:
    # Agregamos un pequeño espacio en blanco para alinear visualmente con la columna 1
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cambiamos selectbox por radio horizontal para forzar la selección por clic
    genero = st.radio("Género:", ["Masculino", "Femenino"], horizontal=True)

st.markdown("---")
st.markdown("#### Hábitos de Riesgo:")

col3, col4 = st.columns(2)
with col3:
    fumador = st.checkbox("Fumador")
    alcohol = st.checkbox("Consumo de Alcohol")
with col4:
    dieta = st.checkbox("Dieta alta en procesados/grasas")

st.markdown("---")

if st.button("Calcular Riesgo", type="primary"):
    # Procesamiento de variables
    genero_val = 1 if genero == "Masculino" else 0
    familia_val = 1 if familia == "Sí" else 0
    fumador_val = 1 if fumador else 0
    alcohol_val = 1 if alcohol else 0
    dieta_val = 1 if dieta else 0

    # Llenar datos con medianas por defecto
    full_data = {col: [df_numeric[col].median()] for col in columnas_modelo}
    
    # Sobrescribir con los inputs del usuario
    full_data['Edad'] = [edad]
    full_data['Genero'] = [genero_val]
    full_data['HistoFamiliar'] = [familia_val]
    full_data['Fumador'] = [fumador_val]
    full_data['Alcohol'] = [alcohol_val]
    full_data['Dieta'] = [dieta_val]
    
    df_input = pd.DataFrame(full_data)
    df_input = df_input[columnas_modelo]

    # Predicción
    pred = et_model.predict(df_input)[0]
    probs = et_model.predict_proba(df_input)[0]
    prob_riesgo = probs[1] * 100

    # Mostrar resultados
    if pred == 1 or prob_riesgo >= 45: 
        st.error(f"**⚠️ RIESGO CLÍNICO ELEVADO**\n\nProbabilidad: **{prob_riesgo:.1f}%**\n\nSe recomienda evaluación médica.")
    else:
        st.success(f"**✅ RIESGO BAJO**\n\nProbabilidad: **{prob_riesgo:.1f}%**\n\nPerfil clínico favorable.")