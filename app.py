import streamlit as st
import joblib
import numpy as np
import requests
import os

# ======================
# Cargar modelo y scaler
# ======================
modelo = joblib.load("vino_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# Hugging Face API Setup
# ======================
HF_TOKEN = os.getenv("HF_TOKEN")  # lo pondremos en secrets de Streamlit
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

def consultar_huggingface(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.5}
    }
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"⚠️ Error {response.status_code}: No se pudo obtener respuesta."

# ======================
# Función para sugerencias
# ======================
def generar_sugerencia(pred):
    if pred == "Bueno":
        return "Mantener el proceso actual y controlar variaciones en alcohol y acidez."
    else:
        return "Reducir acidez volátil en 0.2 y aumentar alcohol en 0.5 para mejorar calidad."

# ======================
# Interfaz Streamlit
# ======================
st.set_page_config(page_title="Análisis de Vino", layout="centered")
st.title("🍷 Sistema de Análisis de Vino")

# Rol del usuario
rol = st.selectbox("Selecciona tu rol:", ["Operario", "Gerente"])

# Formulario para datos
st.subheader("Ingrese las 11 características del vino")
features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

valores = []
for f in features:
    val = st.number_input(f, value=0.0, format="%.3f")
    valores.append(val)

if st.button("Predecir Calidad"):
    X_scaled = scaler.transform([valores])
    pred_num = modelo.predict(X_scaled)[0]
    pred_label = "Bueno" if pred_num == 1 else "Malo"
    sugerencia = generar_sugerencia(pred_label)

    if rol == "Operario":
        st.json({"prediccion": pred_label, "sugerencia": sugerencia})
    else:
        st.json({
            "datos_ingresados": dict(zip(features, valores)),
            "prediccion": pred_label,
            "sugerencia": sugerencia
        })

# Chat para gerente
if rol == "Gerente":
    st.subheader("💬 Chat con Asistente")
    pregunta = st.text_area("Escribe tu pregunta sobre la calidad del vino o proyecciones")
    if st.button("Enviar pregunta"):
        respuesta = consultar_huggingface(pregunta)
        st.write(respuesta)