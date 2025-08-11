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
HF_TOKEN = os.getenv("HF_TOKEN", "hf_tu_token_aqui")  # Cambiar por tu token para pruebas locales
MODEL_NAME = "tiiuae/falcon-7b-instruct"

def generar_respuesta(pregunta):
    """Llama a la API de Hugging Face para generar respuesta del modelo conversacional."""
    if not HF_TOKEN or HF_TOKEN.startswith("hf_tu_token_aqui"):
        return "⚠️ Token de Hugging Face no configurado correctamente."

    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": f"Pregunta: {pregunta}\nRespuesta:",
        "parameters": {"max_new_tokens": 250, "temperature": 0.7}
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
        elif response.status_code == 401:
            return "⚠️ Token inválido o sin permisos. Revisa tu token de Hugging Face."
        elif response.status_code == 404:
            return "⚠️ Modelo no encontrado en Hugging Face."
        else:
            return f"⚠️ Error {response.status_code}: {response.text}"
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

# Datos de entrada precargados para un vino de mala calidad
caracteristicas = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# Valores precargados de ejemplo (vino de baja calidad)
valores_por_defecto = {
    "fixed acidity": 6.0,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
}

valores = {}
for feature in caracteristicas:
    valores[feature] = st.number_input(
        f"{feature}", 
        value=valores_por_defecto[feature], 
        format="%.3f"
    )

# =========================
# Lógica de predicción
# =========================
if st.button("Predecir"):
    X = np.array(list(valores.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediccion = modelo.predict(X_scaled)[0]

    calidad = "Bueno" if prediccion == 1 else "Malo"

    # ===== Operario =====
    if usuario == "operario" and clave == "operario":
        st.success(f"Diagnóstico: {calidad}")

    # ===== Gerente =====
    elif usuario == "gerente" and clave == "gerente":
        st.subheader("📊 Resultados")
        st.write({
            "datos_ingresados": valores,
            "prediccion": calidad,
            "sugerencia": "Reducir acidez volátil en 0.2 y aumentar alcohol en 0.5."
        })

        st.subheader("💬 Chat del Gerente")
        pregunta = st.text_area(
            "Ingrese su pregunta",
            value="¿Cómo mejorar la calidad del vino según las métricas?"
        )
        if st.button("Enviar pregunta"):
            respuesta = generar_respuesta(pregunta)
            st.write("**Pregunta:**", pregunta)
            st.write("**Respuesta:**", respuesta)

    # ===== Credenciales incorrectas =====
    else:
        st.error("Credenciales incorrectas")
