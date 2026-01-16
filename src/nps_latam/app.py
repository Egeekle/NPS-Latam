import streamlit as st
import requests
import os
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --- Config ---
st.set_page_config(page_title="NPS Latam Portal", page_icon="✈️", layout="wide")

# API URL (default to localhost, but overrideable for Docker)
API_URL = os.getenv("API_URL", "http://0.0.0.0:8000")

# Determine Project Root for direct file access (for Dashboard only)
# In a real microservice architecture, dashboard should fetch data via API, not file system.
# But for this monolithic repo structure, file access is acceptable.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

import sys
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

st.title("✈️ Portal de Experiencia del Cliente")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🤖 Asistente Virtual", "📊 Predicción de Satisfacción", "📈 KPI Dashboard"])

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
                "Gender_bin": 0, "CustomerType_bin": 0, "TypeOfTravel_bin": 0, "Class_Eco": 0, "Class_Eco Plus": 0
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

# --- Tab 3: KPI & Dashboard ---
with tab3:
    st.header("📊 KPI Dashboard & Auditoría")
    
    # Load Logs
    log_file = project_root / "Data" / "chatbot_logs.csv"
    
    if log_file.exists():
        try:
            df_logs = pd.read_csv(log_file)
            
            # Metrics
            st.subheader("Métricas de Chatbot")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Interacciones", len(df_logs))
            
            if "timestamp" in df_logs.columns and not df_logs.empty:
                df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
                last_msg = df_logs["timestamp"].max()
                col2.metric("Última Interacción", str(last_msg))

            # Visualize
            st.subheader("Historial de Consultas")
            st.dataframe(df_logs.sort_index(ascending=False), use_container_width=True)
            
            # Basic Analysis (Mock or Real)
            if not df_logs.empty:
                st.subheader("Distribución de Actividad")
                st.bar_chart(df_logs["timestamp"].dt.hour.value_counts())
                st.caption("Interacciones por hora del día")

        except Exception as e:
            st.error(f"Error cargando logs: {e}")
    else:
        st.warning(f"No se encontraron logs en {log_file}")
        
    st.divider()
    
    # --- Advanced KPI Module ---
    st.subheader("🔍 Análisis Avanzado & Tracking")
    
    col1, col2 = st.columns(2)
    
    # 1. Performance Tracking (from MLflow)
    with col1:
        st.markdown("#### 🎯 Performance del Modelo (MLflow)")
        import mlflow
        try:
            # Set tracking URI to local file store
            mlflow.set_tracking_uri("file://" + str(project_root / "mlruns"))
            runs = mlflow.search_runs(experiment_names=["NPS_Latam_Model_Tracking"])
            
            if not runs.empty:
                latest_run = runs.iloc[0]
                st.write(f"**Run ID:** `{latest_run.run_id[:8]}...`")
                st.write(f"**Fecha:** {latest_run['start_time'].strftime('%Y-%m-%d %H:%M')}")
                
                # Dynamic Metrics Display
                metrics_cols = [c for c in runs.columns if c.startswith("metrics.")]
                for metric in metrics_cols:
                    metric_name = metric.replace("metrics.", "")
                    val = latest_run[metric]
                    st.metric(label=metric_name.replace("_", " ").title(), value=f"{val:.4f}")
            else:
                st.info("No se encontraron experimentos registrados.")
        except Exception as e:
            st.error(f"Error leyendo MLflow: {e}")

    # 2. Sentiment KPI (from Chatbot Logs)
    with col2:
        st.markdown("#### ❤️ Customer Sentiment Index (CSI)")
        if log_file.exists():
            if st.button("Calcular KPIs de Sentimiento"):
                with st.spinner("Analizando feedback reciente..."):
                    try:
                        # Extract user queries (limit to last 20 for better demo)
                        user_texts = df_logs["user_query"].dropna().tolist()
                        
                        # Call Analysis API
                        from src.nps_latam.genai_features import analyze_feedback_batch, calculate_csi
                        
                        # Analyze last 20 interactions
                        analysis_limit = 20
                        results_df = analyze_feedback_batch(user_texts[-analysis_limit:])
                        
                        # Calculate CSI Score using module logic
                        csi_score = calculate_csi(results_df)
                        
                        # Metric Display
                        st.metric("CSI Score (0-100)", f"{csi_score:.1f}", delta=f"{csi_score - 50:.1f} vs Neutral")
                        
                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = csi_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Customer Sentiment Index"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "red"},
                                    {'range': [30, 70], 'color': "lightgray"},
                                    {'range': [70, 100], 'color': "green"}],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': csi_score}}))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Breakdown
                        st.subheader("Deep Dive: Intenciones & Tópicos")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Distribución de Sentimientos:**")
                            st.bar_chart(results_df["sentiment"].value_counts())
                            
                        with col_b:
                            if "intent" in results_df.columns:
                                st.write("**Intenciones Detectadas:**")
                                st.bar_chart(results_df["intent"].value_counts())

                        if "keywords" in results_df.columns:
                            st.write("**Top Tópicos Mencinados:**")
                            # Flatten list of lists
                            all_keywords = [k for sublist in results_df["keywords"] for k in sublist]
                            if all_keywords:
                                top_k = pd.Series(all_keywords).value_counts().head(10)
                                st.bar_chart(top_k)
                            else:
                                st.caption("No se detectaron keywords suficientes.")
                        
                        # Show raw data for verification
                        with st.expander("Ver Datos Procesados"):
                            st.dataframe(results_df)
                        
                    except Exception as e:
                        st.error(f"Error en análisis: {e}")
            else:
                st.caption("Haga clic para analizar el sentimiento de las últimas 20 interacciones.")
        else:
            st.warning("Sin datos para analizar.")

    st.divider()
    
    st.info("Dashboard MLflow (Docker): http://localhost:5001 | (Local): Run `mlflow ui`")

# --- Sidebar info ---
st.sidebar.markdown(f"**Estado del Sistema**")
try:
    health = requests.get(f"{API_URL}/health", timeout=1).json()
    st.sidebar.success(f"API Online: {health.get('status')}")
    st.sidebar.markdown(f"- Modelo Cargado: {'✅' if health.get('model_loaded') else '❌'}")
    st.sidebar.markdown(f"- Chatbot Cargado: {'✅' if health.get('chatbot_loaded') else '❌'}")
except:
    st.sidebar.error("API Offline")
    st.sidebar.info("backend no detectado en localhost:8000")
