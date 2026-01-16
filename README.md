---
title: NPS Latam
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Proyecto de Análisis Predictivo de Negocios: Satisfacción de Pasajeros de Aerolíneas

Este proyecto tiene como objetivo desarrollar un modelo predictivo para estimar la satisfacción de los pasajeros de aerolíneas, permitiendo a las empresas identificar áreas clave de mejora en sus servicios. El análisis se estructura en dos fases principales: ingeniería de datos y modelado predictivo.

## Estructura del Proyecto

El flujo de trabajo se divide en dos fases secuenciales:

1.  **Fase 1 (Trabajo Práctico)**: Enfocada en la preparación, limpieza y transformación de datos.
2.  **Fase 2 (Trabajo Final)**: Enfocada en la selección de características, entrenamiento de modelos, ajuste de hiperparámetros y evaluación final.

---

## 1. Ingeniería de Datos (Fase 1)

En esta primera fase se procesa el dataset original `Satisfacción de pasajeros.csv` para prepararlo para el modelado.

### Limpieza y Preprocesamiento
*   **Traducción**: Renombrado de columnas de inglés a español para facilitar la interpretación (ej. `Flight Distance` -> `Distancia_Vuelo`).
*   **Manejo de Valores Atípicos**: Se detectaron outliers en la variable `Distancia_Vuelo`. Se optó por la imputación utilizando la mediana para reducir el impacto de valores extremos sin perder datos.
*   **Eliminación de Columnas**: Se descartaron columnas irrelevantes o redundantes como `ID`, `Retraso_Salida_mim`, y `Retraso_Llegada_mim`.

### Ingeniería de Características (Feature Engineering)
Se crearon nuevas variables para capturar mejor el comportamiento de los pasajeros:
*   **Agregaciones de Servicios**: Se generaron estadísticas descriptivas (Media, Mínimo, Máximo, Varianza) basadas en las columnas de puntuación de servicios (ej. `Service_Mean`, `Service_Var`) para resumir la percepción general del cliente.
*   **Binning (Categorización)**:
    *   `Age_Bin`: Categorización de la edad en grupos.
    *   `Distance_Bin`: Categorización de la distancia de vuelo.

### Transformación
*   **Codificación (Encoding)**: Se aplicó *One-Hot Encoding* (variables dummy) a variables categóricas nominales como `Género`, `Tipo de Cliente`, `Tipo de Viaje` y `Clase`.
*   **Escalado**: Se utilizó `StandardScaler` de Scikit-Learn para normalizar las variables numéricas, asegurando que todas tengan media 0 y desviación estándar 1.
*   **Target**: La variable objetivo `Satisfacción` se binarizó (1: Satisfecho, 0: Neutral/Insatisfecho).

**Salida**: El dataset procesado se guarda como `airline_satisfaction_transformed_clean.csv`.

---

## 2. Modelado Predictivo (Fase 2)

En esta segunda fase se utilizan los datos procesados para entrenar y validar múltiples modelos de clasificación.

### Selección de Características
Se implementó **RFECV** (Recursive Feature Elimination con Cross-Validation) utilizando un `RandomForestClassifier` como estimador base.
*   **Resultado**: Se seleccionaron **27 características** óptimas de las 34 iniciales, maximizando la métrica ROC-AUC.

### Modelos Evaluados
Se probaron y ajustaron los siguientes algoritmos:

1.  **Regresión Logística**:
    *   *Base vs. Ajustado*: Se utilizó `GridSearchCV` para optimizar `C`, `penalty` (L1/L2) y `solver`.
    *   *Rendimiento*: Proporcionó una línea base sólida con un Accuracy ~89%.

2.  **Árbol de Decisión**:
    *   *Ajuste*: Se optimizaron `criterion` (gini/entropy), `max_depth`, `min_samples_leaf` y `min_samples_split`.
    *   *Rendimiento*: Mejoró significativamente respecto a la regresión logística (Accuracy ~94.8%).

3.  **Random Forest (Mejor Modelo)**:
    *   *Ajuste*: Se evaluaron múltiples estimadores y profundidades.
    *   *Resultados*: Mostró el mejor desempeño general con gran estabilidad.
    *   **Métricas Finales (Validación)**:
        *   **AUC**: ~0.9928
        *   **F1-Score**: ~0.9521
        *   **Accuracy**: ~95.91%

4.  **XGBoost**:
    *   *Ajuste*: Optimización bayesiana/aleatoria de `learning_rate`, `n_estimators`, `max_depth`, `subsample`, etc.
    *   *Resultados*: Rendimiento muy cercano a Random Forest, siendo una alternativa muy competitiva.

### Conclusiones del Modelado
*   **Random Forest** y **XGBoost** fueron los modelos superiores, alcanzando métricas de excelencia (AUC > 0.99).
*   El modelo es altamente robusto y generalizable, como lo demuestra la baja desviación estándar en la validación cruzada (CV Std ~0.002).
*   Las características de servicios agregadas demostraron ser predictoras importantes.

---

## 3. Productivización e Implementación (Trabajo Final Integrador)

En esta fase final, se transformó el modelo estático en una **solución de software completa**, integrando Inteligencia Artificial Generativa (LMMs), MLOps y un Dashboard interactivo.

### Funcionalidades Clave

1.  **Asistente Virtual Inteligente (Chatbot)**:
    *   **Tecnología**: API de **Gemini 2.5 Flash** (Google DeepMind).
    *   **Propósito**: Atender consultas naturales de los pasajeros (ej. "¿Tienen comida vegetariana?").
    *   **Features**: Registro automático de conversiones y extracción de contexto.

2.  **Dashboard de KPIs (Streamlit)**:
    *   Interfaz web interactiva para visualizar métricas de negocio.
    *   **KPI Principal (CSI - Customer Sentiment Index)**: Un indicador de 0 a 100 que mide la satisfacción en tiempo real basado en el análisis de sentimiento de las conversaciones del chatbot.
    *   *Visualización*: Gráfico de "Gauge" (velocímetro) que alerta si el sentimiento es Positivo (Verde), Neutral (Gris) o Negativo (Rojo).

3.  **MLOps y Tracking (MLflow)**:
    *   Implementación de **MLflow** para rastrear experimentos de entrenamiento.
    *   Registro de métricas clave (Accuracy, F1-Score, AUC) y parámetros del modelo para auditoría continua.

4.  **Arquitectura de Despliegue (Docker)**:
    *   Sistema unificado en un contenedor "Monolito" optimizado para demostraciones Robustas.
    *   **FastAPI**: Backend de alto rendimiento para servir el modelo y el chatbot.
    *   **Streamlit**: Frontend amigable para el usuario final.

---

## 🚀 Guía de Ejecución Rápida (Docker)

El proyecto está dockerizado para garantizar la reproducibilidad. Siga estos pasos para ejecutar toda la plataforma:

**Requisitos Previos**
*   Docker & Docker Compose instalados.
*   Una API Key de Google (Gemini) configurada en un archivo `.env` (`GOOGLE_API_KEY=...`).

**Comando de Inicio**
Ejecute el siguiente comando en la raíz del proyecto:

```bash
docker-compose up --build
```

**Acceso a la Plataforma**
Una vez iniciado el contenedor, acceda a los servicios en su navegador:

*   **💻 Portal de Cliente (Frontend)**: [http://localhost:8501](http://localhost:8501)
    *   *Interactúe con el Chatbot, realice Predicciones y vea el Dashboard de KPIs.*
*   **📊 MLflow Tracking (Experimentos)**: [http://localhost:5001](http://localhost:5001)
*   **⚙️ API Backend (Documentación)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 💡 Ejemplos de Interacción con el Chatbot

Una vez en el portal (Tab: *Asistente Virtual*), intente las siguientes preguntas para validar el análisis de sentimiento y el **CSI**:

1.  **Neutral/Informativa**:
    > *"¿Cuál es el límite de peso para el equipaje de mano?"*
2.  **Negativa (Queja)**:
    > *"Estoy muy molesto, mi vuelo se retrasó 3 horas y nadie me dio información."*
3.  **Positiva (Felicitación)**:
    > *"¡Me encantó el servicio a bordo! La comida estaba deliciosa y el asiento muy comodó."*
4.  **Solicitud Especial**:
    > *"¿Puedo llevar a mi mascota en cabina en un vuelo internacional?"*

*Nota: Después de interactuar, vaya a la pestaña "KPI Dashboard" y actualice las métricas para ver cómo sus mensajes impactan el Customer Sentiment Index.*

---

## Requisitos y Configuración (Entorno Local Python)

El proyecto fue desarrollado originalmente en un entorno de **Google Colab**. Para reproducirlo localmente (sin Docker), se requieren las siguientes bibliotecas principales (ver `pyproject.toml` para detalle completo):

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## Autores (Grupo 4)
*   Avendaño Alvarez, Elsida Janiria
*   Cordova Peña, Hitalo Bernabé
*   García Cárdenas, Ramiro Sebastián
*   Reyes Zuñiga, Oscar Aldahir
*   Umiña Navia, Luis Angel
