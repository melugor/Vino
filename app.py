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
MODEL_NAME = "google/flan-t5-base"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# 🔹 Cargar token de forma segura desde Streamlit Secrets
HF_TOKEN = st.secrets["HF_TOKEN"]
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def generar_respuesta(pregunta):
    """Genera una respuesta usando el modelo de Hugging Face"""
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
# Interfaz en Streamlit
# =========================
st.set_page_config(page_title="Predicción Calidad de Vino", layout="centered")

st.title("🍷 Predicción de Calidad del Vino")
st.write("Ingrese sus credenciales para acceder.")

# =========================
# Login
# =========================
usuario = st.text_input("Usuario")
clave = st.text_input("Clave", type="password")

# Datos precargados para vino de mala calidad
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

# Crear inputs con valores por defecto
valores = {}
for feature, default in datos_defecto.items():
    valores[feature] = st.number_input(f"{feature}", value=float(default), format="%.2f")

# =========================
# Lógica según usuario
# =========================
if usuario and clave:
    if usuario == "operario" and clave == "operario":
        if st.button("Predecir"):
            X = np.array(list(valores.values())).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prediccion = modelo.predict(X_scaled)[0]
            calidad = "Bueno" if prediccion == 1 else "Malo"
            st.success(f"Diagnóstico: {calidad}")

    elif usuario == "gerente" and clave == "gerente":
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

    else:
        st.error("Credenciales incorrectas")
