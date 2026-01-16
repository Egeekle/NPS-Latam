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

## Requisitos y Configuración

El proyecto fue desarrollado en un entorno de **Google Colab**. Para reproducirlo localmente, se requieren las siguientes bibliotecas principales:

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
