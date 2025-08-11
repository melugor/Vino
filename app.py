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
# Configuración del modelo (sin token)
# =========================
MODEL_NAME = "google/flan-t5-base"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def generar_respuesta(pregunta):
    payload = {
        "inputs": f"Responde de forma clara y breve: {pregunta}",
        "parameters": {"max_new_tokens": 250, "temperature": 0.7}
    }
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            else:
                return "⚠️ No se pudo interpretar la respuesta del modelo."
        else:
            return f"⚠️ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Error: {e}"

# =========================
# Configuración página
# =========================
st.set_page_config(page_title="Predicción Calidad de Vino", layout="centered")

# Inicializar variables de sesión
if "usuario" not in st.session_state:
    st.session_state.usuario = None
if "chat_historial" not in st.session_state:
    st.session_state.chat_historial = []
if "logueado" not in st.session_state:
    st.session_state.logueado = False

st.title("🍷 Predicción de Calidad del Vino")

# =========================
# Pantalla de login
# =========================
if not st.session_state.logueado:
    usuario = st.text_input("Usuario")
    clave = st.text_input("Clave", type="password")
    if st.button("Ingresar"):
        if (usuario == "operario" and clave == "operario") or (usuario == "gerente" and clave == "gerente"):
            st.session_state.usuario = usuario
            st.session_state.logueado = True
        else:
            st.error("Credenciales incorrectas")

# =========================
# Interfaz de Operario
# =========================
elif st.session_state.usuario == "operario":
    st.success("Bienvenido Operario 👷‍♂️")
    
    caracteristicas = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    valores = {feat: st.number_input(feat, value=0.0, format="%.2f") for feat in caracteristicas}

    if st.button("Predecir"):
        X = np.array(list(valores.values())).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediccion = modelo.predict(X_scaled)[0]
        calidad = "Bueno" if prediccion == 1 else "Malo"
        st.success(f"Diagnóstico: {calidad}")

# =========================
# Interfaz de Gerente
# =========================
elif st.session_state.usuario == "gerente":
    st.success("Bienvenido Gerente 👨‍💼")
    
    caracteristicas = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
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
    valores = {feat: st.number_input(feat, value=valores_por_defecto[feat], format="%.3f") for feat in caracteristicas}

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

    # Chat del gerente (siempre visible)
    st.subheader("💬 Chat del Gerente")
    pregunta = st.text_area("Ingrese su pregunta")
    if st.button("Enviar pregunta"):
        if pregunta.strip():
            respuesta = generar_respuesta(pregunta)
            st.session_state.chat_historial.append((pregunta, respuesta))

    # Mostrar historial de chat
    for idx, (q, r) in enumerate(st.session_state.chat_historial):
        st.write(f"**Pregunta {idx+1}:** {q}")
        st.write(f"**Respuesta:** {r}")

# Botón para cerrar sesión
if st.session_state.logueado:
    if st.button("Cerrar sesión"):
        st.session_state.clear()
