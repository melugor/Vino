import streamlit as st
import joblib
import numpy as np
import os
import requests

# =========================
# Cargar modelo y escalador
# =========================
modelo = joblib.load("vino_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Configuración Hugging Face
# =========================
HF_TOKEN = os.getenv("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")  # Variable de entorno en Streamlit Cloud
MODEL_NAME = "google/flan-t5-base"

def generar_respuesta(pregunta):
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": pregunta,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            else:
                return "⚠️ No se pudo interpretar la respuesta del modelo."
        elif response.status_code == 404:
            return "⚠️ Modelo no encontrado en Hugging Face."
        else:
            return f"⚠️ Error {response.status_code}: No se pudo obtener respuesta."
    except Exception as e:
        return f"⚠️ Error: {e}"

# =========================
# Interfaz en Streamlit
# =========================
st.set_page_config(page_title="Predicción Calidad de Vino", layout="centered")

st.title("🍷 Predicción de Calidad del Vino")
st.write("Ingrese sus credenciales para acceder.")

usuario = st.text_input("Usuario")
clave = st.text_input("Clave", type="password")

# Credenciales de ejemplo
if usuario == "operario" and clave == "operario":
    st.header("🔧 Panel del Operario")
    
    # Datos por defecto para vino de mala calidad
    valores_defecto = {
        "fixed acidity": 7.0, "volatile acidity": 1.0, "citric acid": 0.0, "residual sugar": 1.0,
        "chlorides": 0.1, "free sulfur dioxide": 5.0, "total sulfur dioxide": 15.0, "density": 1.003,
        "pH": 3.0, "sulphates": 0.3, "alcohol": 8.0
    }

    valores = {}
    for feature, default in valores_defecto.items():
        valores[feature] = st.number_input(f"{feature}", value=default, format="%.2f")

    if st.button("Predecir calidad"):
        X = np.array(list(valores.values())).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediccion = modelo.predict(X_scaled)[0]
        calidad = "Bueno" if prediccion == 1 else "Malo"
        st.success(f"Diagnóstico: {calidad}")

elif usuario == "gerente" and clave == "gerente":
    st.header("📊 Panel del Gerente")

    # Datos por defecto para vino de mala calidad
    valores_defecto = {
        "fixed acidity": 7.0, "volatile acidity": 1.0, "citric acid": 0.0, "residual sugar": 1.0,
        "chlorides": 0.1, "free sulfur dioxide": 5.0, "total sulfur dioxide": 15.0, "density": 1.003,
        "pH": 3.0, "sulphates": 0.3, "alcohol": 8.0
    }

    valores = {}
    for feature, default in valores_defecto.items():
        valores[feature] = st.number_input(f"{feature}", value=default, format="%.2f")

    if st.button("Predecir calidad"):
        X = np.array(list(valores.values())).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediccion = modelo.predict(X_scaled)[0]
        calidad = "Bueno" if prediccion == 1 else "Malo"

        st.write({
            "datos_ingresados": valores,
            "prediccion": calidad,
            "sugerencia": "Reducir acidez volátil en 0.2 y aumentar alcohol en 0.5."
        })

    # Chat persistente con session_state
    st.subheader("💬 Chat del Gerente")
    if "preguntas" not in st.session_state:
        st.session_state["preguntas"] = []
    if "respuestas" not in st.session_state:
        st.session_state["respuestas"] = []

    pregunta = st.text_area("Ingrese su pregunta", value="", key="pregunta_input")
    if st.button("Enviar pregunta"):
        if pregunta.strip():
            respuesta = generar_respuesta(pregunta)
            st.session_state["preguntas"].append(pregunta)
            st.session_state["respuestas"].append(respuesta)

    # Mostrar historial
    for q, r in zip(st.session_state["preguntas"], st.session_state["respuestas"]):
        st.markdown(f"**Pregunta:** {q}")
        st.markdown(f"**Respuesta:** {r}")
        st.write("---")

elif usuario or clave:  # Si escribió algo pero no coincide
    st.error("Credenciales incorrectas")
