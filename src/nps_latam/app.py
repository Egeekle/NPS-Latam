import streamlit as st
import requests
import os
import json

# --- Config ---
st.set_page_config(page_title="NPS Latam Portal", page_icon="✈️", layout="wide")

# API URL (default to localhost, but overrideable for Docker)
API_URL = os.getenv("API_URL", "http://0.0.0.0:8000")

st.title("✈️ Portal de Experiencia del Cliente")

# --- Tabs ---
tab1, tab2 = st.tabs(["🤖 Asistente Virtual", "📊 Predicción de Satisfacción"])

# --- Tab 1: Chatbot ---
with tab1:
    st.header("Asistente Virtual")
    st.write("Consulta sobre tu vuelo o servicios.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle Input
    if prompt := st.chat_input("Escribe tu consulta aquí..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Conectando con el servidor..."):
                    resp = requests.post(f"{API_URL}/chat", json={"message": prompt})
                    if resp.status_code == 200:
                        full_response = resp.json().get("response", "No response content.")
                    else:
                        full_response = f"Error del servidor: {resp.status_code} - {resp.text}"
            except Exception as e:
                full_response = f"No se pudo conectar con el API: {str(e)}"

            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.button("Borrar Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Tab 2: Prediction ---
with tab2:
    st.header("Predicción de Satisfacción")
    st.write("Ingrese los datos del pasajero para estimar su satisfacción.")

    # Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad", min_value=0, max_value=100, value=30)
            flight_dist = st.number_input("Distancia de Vuelo (km)", min_value=0, value=500)
            wifi = st.slider("Wifi a bordo (1-5)", 1, 5, 3)
            food = st.slider("Comida y Bebida (1-5)", 1, 5, 3)
            comfort = st.slider("Comodidad Asiento (1-5)", 1, 5, 3)
        
        with col2:
            seat_comfort = st.slider("Confort del asiento (1-5)", 1, 5, 3)
            entertainment = st.slider("Entretenimiento (1-5)", 1, 5, 3)
            onboard_service = st.slider("Servicio a bordo (1-5)", 1, 5, 3)
            leg_room = st.slider("Espacio para piernas (1-5)", 1, 5, 3)
            baggage = st.slider("Manejo de equipaje (1-5)", 1, 5, 3)

        # Submit
        submitted = st.form_submit_button("Predecir")
        
        if submitted:
            # Construct feature dictionary matching the model's expected columns
            # Note: The model behaves based on the training data columns.
            # We map form inputs to likely column names. 
            # Ideally, we should fetch the schema from the API, but hardcoding for demo is fine.
            features = {
                "Edad": age,
                "Distancia_Vuelo": flight_dist,
                "Wifi_a_bordo": wifi,
                "Comida_Bebida": food,
                "Comodidad_Asiento": comfort,
                "Entretenimiento": entertainment,
                "Servicio_a_bordo": onboard_service,
                "Espacio_Piernas": leg_room,
                "Manejo_Equipaje": baggage,
                # Add dummy values for categorical binaries or others if strictly required by model pipeline
                # Assuming model handles missing/zeros gracefully via reindexing in API
                "Gender_bin": 0, 
                "CustomerType_bin": 0,
                "TypeOfTravel_bin": 0,
                "Class_Eco": 0,
                "Class_Eco Plus": 0
            }
            
            try:
                resp = requests.post(f"{API_URL}/predict", json={"data": features})
                if resp.status_code == 200:
                    result = resp.json()
                    label = result["label"]
                    prob = result["probability"]
                    
                    st.success(f"Predicción: **{label}**")
                    st.metric("Probabilidad de Satisfacción", f"{prob:.2%}")
                else:
                    st.error(f"Error en predicción: {resp.text}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")

# --- Sidebar info ---
st.sidebar.markdown(f"**Estado del Sistema**")
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.success(f"API Online: {health.get('status')}")
    st.sidebar.markdown(f"- Modelo Cargado: {'✅' if health.get('model_loaded') else '❌'}")
    st.sidebar.markdown(f"- Chatbot Cargado: {'✅' if health.get('chatbot_loaded') else '❌'}")
except:
    st.sidebar.error("API Offline")
    st.sidebar.info("Asegúrate de ejecutar el servidor backend.")
