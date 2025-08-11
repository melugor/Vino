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
HF_TOKEN = os.getenv("HF_TOKEN")  # Variable de entorno en Streamlit Cloud
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

# Datos de entrada
caracteristicas = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

valores = {}
for feature in caracteristicas:
    valores[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

if st.button("Predecir"):
    X = np.array(list(valores.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediccion = modelo.predict(X_scaled)[0]

    calidad = "Bueno" if prediccion == 1 else "Malo"

    if usuario == "operario" and clave == "operario":
        st.success(f"Diagnóstico: {calidad}")
    elif usuario == "gerente" and clave == "gerente":
        st.subheader("📊 Resultados")
        st.write({
            "datos_ingresados": valores,
            "prediccion": calidad,
            "sugerencia": "Reducir acidez volátil en 0.2 y aumentar alcohol en 0.5."
        })

        st.subheader("💬 Chat del Gerente")
        pregunta = st.text_area("Ingrese su pregunta", value="¿Cómo mejorar la calidad del vino según las métricas?")
        if st.button("Enviar pregunta"):
            respuesta = generar_respuesta(pregunta)
            st.write("**Respuesta:**", respuesta)
    else:
        st.error("Credenciales incorrectas")
