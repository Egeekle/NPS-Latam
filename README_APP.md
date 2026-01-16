# Proyecto: Sistema Inteligente de Satisfacción del Cliente (NPS Latam)

## Planteamiento del Problema
La industria aérea enfrenta el desafío constante de mantener altos niveles de satisfacción del cliente en un entorno competitivo. Entender los factores que influyen en la lealtad y satisfacción del pasajero es crucial. Actualmente, el análisis puede limitarse a encuestas estructuradas post-vuelo, perdiendo la riqueza de la retroalimentación inmediata y no estructurada. El problema central es predecir y mejorar el Net Promoter Score (NPS) o la satisfacción general utilizando tanto datos históricos como interacciones en tiempo real.

## Negocio y Objetivos
El objetivo principal es desarrollar una aplicación analítica integral que permita:
1.  **Predecir la Satisfacción:** Estimar la probabilidad de que un cliente esté satisfecho basándose en sus características de viaje y calificaciones de servicio.
2.  **Identificar Drivers Clave:** Determinar qué servicios (Wifi, Comida, Comodidad, etc.) tienen mayor impacto en la satisfacción.
3.  **Interacción Inteligente (Chatbot):** Implementar un **Chatbot** integrado en la app para que el cliente interactúe, realice consultas y provea feedback cualitativo en lenguaje natural.
4.  **Feature Engineering con GenAI:** Utilizar Inteligencia Artificial Generativa para procesar los logs de las consultas del chatbot, extrayendo nuevas características (sentimiento, temas recurrentes, intención) que enriquezcan el modelo predictivo.
5.  **Monitoreo de Drift:** Establecer un sistema de monitoreo continuo para detectar **Data Drift** (cambios en la distribución de los datos de entrada) y **Concept Drift** (cambios en la relación entre variables y la satisfacción), asegurando la vigencia del modelo.

## Hipótesis
La integración de características derivadas del análisis de texto no estructurado (proveniente del Chatbot y procesado con GenAI), combinada con las métricas tradicionales de servicio, mejorará significativamente la precisión del modelo predictivo de satisfacción del cliente en comparación con el uso exclusivo de datos estructurado. Además, el monitoreo activo de drift permitirá acciones correctivas tempranas para mantener el rendimiento del modelo.

## Acciones con los Entregables
-   **Despliegue de App:** Poner en producción una interfaz web (Streamlit o similar) que incluya el formulario de predicción y la interfaz del chatbot.
-   **Mejora Continua:** Utilizar los reportes de Drift para decidir cuándo re-entrenar el modelo con nuevos datos recolectados.
-   **Estrategia de Servicio:** Proveer insights a la gerencia sobre qué áreas de servicio necesitan inversión inmediata basada en la importancia de las características.

## Acceso a Datos y Tipo de Datos

### Datos Estructurados
-   **Fuente:** Archivos CSV (ej. `synthetic_nps_latam.csv`).
-   **Contenido:**
    -   *Demográficos:* Género, Edad, Tipo de Cliente.
    -   *Viaje:* Tipo de Viaje, Clase, Distancia.
    -   *Servicios (Escala 1-5):* Wifi, Comodidad, Comida, Limpieza, etc.
    -   *Target:* Satisfacción / Lealtad.

### Datos No Estructurados
-   **Fuente:** Logs de interacción del Chatbot.
-   **Contenido:** Texto libre de consultas de usuarios, quejas, felicitaciones y comentarios sobre la experiencia de vuelo.

## Tipo de Solución a Elaborar
Una **Aplicación Web Inteligente** que combina:
1.  **Backend de Modelado:** Pipeline de entrenamiento y predicción (Scikit-learn/XGBoost).
2.  **Módulo GenAI:** Procesamiento de NLP para enriquecimiento de datos (LLM integration).
3.  **Frontend Interactivo:** Interfaz para el usuario final y dashboard de monitoreo para los científicos de datos/gerentes.
4.  **Sistema de Observabilidad:** Monitoreo de features y drift del modelo.
