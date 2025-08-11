import streamlit as st
import joblib
import numpy as np
import requests

# ===== CONFIGURACIÓN =====
HF_TOKEN = st.secrets["HF_TOKEN"]  # Se guarda como secreto en Streamlit Cloud
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# ===== CARGAR MODELO Y SCALER =====
modelo = joblib.load("vino_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===== FUNCIONES =====
def predecir_calidad(datos):
    datos_np = np.array(datos).reshape(1, -1)
    datos_scaled = scaler.transform(datos_np)
    pred = modelo.predict(datos_scaled)[0]
    return "Bueno" if pred == 1 else "Malo"

def sugerencia_mejora(prediccion):
    if prediccion == "Bueno":
        return "Mantener el nivel de alcohol y acidez; revisar consistencia en la producción."
    else:
        return "Reducir acidez volátil en 0.2 y aumentar alcohol en 0.5."

def consultar_huggingface(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"⚠️ Error {response.status_code}: No se pudo obtener respuesta."

# ===== INTERFAZ =====
st.set_page_config(page_title="Análisis de Vino", layout="centered")
st.title("🍷 Análisis y Recomendaciones de Calidad del Vino")

rol = st.selectbox("Selecciona tu rol:", ["Operario", "Gerente"])

st.subheader("📥 Ingresar mediciones del vino")
campos = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

valores = []
for campo in campos:
    valores.append(st.number_input(campo, value=0.0))

if st.button("Analizar calidad"):
    pred = predecir_calidad(valores)
    sug = sugerencia_mejora(pred)

    if rol == "Operario":
        st.json({"prediccion": pred, "sugerencia": sug})
    else:  # Gerente
        st.json({
            "datos_ingresados": dict(zip(campos, valores)),
            "prediccion": pred,
            "sugerencia": sug
        })

if rol == "Gerente":
    st.subheader("💬 Consulta sobre métricas e insights")
    pregunta = st.text_area("Escribe tu pregunta:")
    if st.button("Enviar pregunta"):
        if pregunta.strip():
            respuesta = consultar_huggingface(
                f"Responde de forma sencilla en español sobre métricas, insights o proyecciones del negocio de vinos. Pregunta: {pregunta}"
            )
            st.write("**Respuesta del asistente:**")
            st.write(respuesta)
        else:
            st.warning("Por favor escribe una pregunta.")
