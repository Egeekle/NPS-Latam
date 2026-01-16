import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_rows=1000, seed=42):
    np.random.seed(seed)
    
    # --- 1. Basic Categorical Columns ---
    genero = np.random.choice(['Male', 'Female'], num_rows)
    tipo_cliente = np.random.choice(['Loyal Customer', 'disloyal Customer'], num_rows, p=[0.8, 0.2])
    tipo_viaje = np.random.choice(['Business travel', 'Personal Travel'], num_rows, p=[0.7, 0.3])
    clase = np.random.choice(['Business', 'Eco', 'Eco Plus'], num_rows, p=[0.5, 0.4, 0.1])
    
    # --- 2. Numerical Columns ---
    edad = np.random.randint(7, 86, num_rows)
    distancia_vuelo = np.random.randint(100, 5000, num_rows)
    
    # --- 3. Ordinal Service Columns (Likert 1-5) ---
    # Core columns from snippet
    wifi = np.random.randint(1, 6, num_rows)
    comodidad_horario = np.random.randint(1, 6, num_rows)
    facilidad_reserva = np.random.randint(1, 6, num_rows)
    ubicacion_puerta = np.random.randint(1, 6, num_rows)
    
    # Assumed standard columns (the "..." part)
    comida_bebida = np.random.randint(1, 6, num_rows)
    embarque_online = np.random.randint(1, 6, num_rows)
    comodidad_asiento = np.random.randint(1, 6, num_rows)
    entretenimiento = np.random.randint(1, 6, num_rows)
    servicio_abordo = np.random.randint(1, 6, num_rows)
    espacio_piernas = np.random.randint(1, 6, num_rows)
    manejo_equipaje = np.random.randint(1, 6, num_rows)
    servicio_checkin = np.random.randint(1, 6, num_rows)
    servicio_vuelo = np.random.randint(1, 6, num_rows)
    limpieza = np.random.randint(1, 6, num_rows)
    
    # Create DataFrame with raw data first
    df = pd.DataFrame({
        'Genero': genero,
        'Tipo_Cliente': tipo_cliente,
        'Edad': edad,
        'Tipo_Viaje': tipo_viaje,
        'Clase': clase,
        'Distancia_Vuelo': distancia_vuelo,
        'Wifi_a_bordo': wifi,
        'Comodidad_Horario': comodidad_horario,
        'Facilidad_Reserva': facilidad_reserva,
        'Ubicacion_Puerta': ubicacion_puerta,
        # Assumed columns
        'Comida_Bebida': comida_bebida,
        'Embarque_Online': embarque_online,
        'Comodidad_Asiento': comodidad_asiento,
        'Entretenimiento': entretenimiento,
        'Servicio_Abordo': servicio_abordo,
        'Espacio_Piernas': espacio_piernas,
        'Manejo_Equipaje': manejo_equipaje,
        'Servicio_Checkin': servicio_checkin,
        'Servicio_Vuelo': servicio_vuelo,
        'Limpieza': limpieza
    })
    
    # --- 4. Derived Features ---
    
    # TypeOfTravel_bin
    df['TypeOfTravel_bin'] = (df['Tipo_Viaje'] == 'Business travel').astype(int)
    
    # Class_Eco, Class_Eco Plus (One-hot manual)
    df['Class_Eco'] = (df['Clase'] == 'Eco')
    df['Class_Eco Plus'] = (df['Clase'] == 'Eco Plus')
    
    # Service Stats
    service_cols = [
        'Wifi_a_bordo', 'Comodidad_Horario', 'Facilidad_Reserva', 'Ubicacion_Puerta',
        'Comida_Bebida', 'Embarque_Online', 'Comodidad_Asiento', 'Entretenimiento',
        'Servicio_Abordo', 'Espacio_Piernas', 'Manejo_Equipaje', 'Servicio_Checkin', 
        'Servicio_Vuelo', 'Limpieza'
    ]
    
    df['Service_Mean'] = df[service_cols].mean(axis=1)
    df['Service_Min'] = df[service_cols].min(axis=1)
    df['Service_Max'] = df[service_cols].max(axis=1)
    df['Service_Var'] = df[service_cols].var(axis=1)
    
    # Binning
    # Simple quantiles for demo purposes, matching the float format in snippet
    df['Age_Bin'] = pd.qcut(df['Edad'], q=5, labels=False).astype(float)
    df['Distance_Bin'] = pd.qcut(df['Distancia_Vuelo'], q=5, labels=False).astype(float)
    
    # --- 5. Target Generation ---
    # Logic: Higher satisfaction scores -> Higher probability of target=1 (satisfied?)
    # Note: In snippet, 0 seems to be "neutral or dissatisfied" and 1 "satisfied" (or vice versa, typically 1 is target)
    # Let's assume linear relationship with noise
    prob = (df['Service_Mean'] / 5.0) * 0.8 + 0.1 # scaled roughly 0.1 to 0.9
    # Add some randomness based on loyal/disloyal
    prob += np.where(df['Tipo_Cliente'] == 'Loyal Customer', 0.1, -0.1)
    prob = np.clip(prob, 0, 1)
    
    df['target'] = np.random.binomial(1, prob)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_rows=1000)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'synthetic_nps_latam.csv')
    
    df.to_csv(output_path, index_label='id') # index_label='id' to match implicit 0,1,2,3 index in snippet
    print(f"Data saved to {output_path}")
    print(df.head())
