import streamlit as st
import joblib
import numpy as np
import requests

# =========================
# Cargar modelo y escalador
# =========================
modelo = joblib.load("vino_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Configuración Hugging Face
# =========================
MODEL_NAME = "facebook/blenderbot-400M-distill"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HF_TOKEN = st.secrets["HF_TOKEN"]  # Token desde secrets
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generar_respuesta(pregunta):
    """Genera una respuesta usando Hugging Face"""
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
        elif response.status_code == 401:
            return "⚠️ Error 401: Token inválido o sin permisos."
        else:
            return f"⚠️ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# =========================
# Configuración de página
# =========================
st.set_page_config(page_title="Predicción Calidad de Vino", layout="centered")
st.title("🍷 Predicción de Calidad del Vino")

# =========================
# Control de sesión
# =========================
if "rol" not in st.session_state:
    st.session_state["rol"] = None

# =========================
# Login
# =========================
if st.session_state["rol"] is None:
    st.write("Ingrese sus credenciales para acceder.")
    usuario = st.text_input("Usuario")
    clave = st.text_input("Clave", type="password")

    if st.button("Iniciar sesión"):
        if usuario == "operario" and clave == "operario":
            st.session_state["rol"] = "operario"
        elif usuario == "gerente" and clave == "gerente":
            st.session_state["rol"] = "gerente"
        else:
            st.error("Credenciales incorrectas")

# =========================
# Datos precargados para análisis
# =========================
datos_defecto = {
    "fixed acidity": 9.5,
    "volatile acidity": 0.8,
    "citric acid": 0.05,
    "residual sugar": 1.9,
    "chlorides": 0.07,
    "free sulfur dioxide": 5.0,
    "total sulfur dioxide": 15.0,
    "density": 0.997,
    "pH": 3.2,
    "sulphates": 0.5,
    "alcohol": 9.0
}

# =========================
# Vista Operario
# =========================
if st.session_state["rol"] == "operario":
    st.subheader("Panel del Operario")
    valores = {f: st.number_input(f, value=float(v), format="%.2f") for f, v in datos_defecto.items()}

    if st.button("Predecir"):
        X = np.array(list(valores.values())).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediccion = modelo.predict(X_scaled)[0]
        calidad = "Bueno" if prediccion == 1 else "Malo"
        st.success(f"Diagnóstico: {calidad}")

# =========================
# Vista Gerente
# =========================
elif st.session_state["rol"] == "gerente":
    st.subheader("Panel del Gerente")
    valores = {f: st.number_input(f, value=float(v), format="%.2f") for f, v in datos_defecto.items()}

    if st.button("Predecir"):
        X = np.array(list(valores.values())).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediccion = modelo.predict(X_scaled)[0]
        calidad = "Bueno" if prediccion == 1 else "Malo"

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
