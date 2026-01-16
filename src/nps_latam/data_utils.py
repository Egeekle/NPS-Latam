import pandas as pd
import os
try:
    from nps_latam import config
except ImportError:
    # Fallback if run directly from root without package context
    import config

def load_processed_dataset(path=None):
    """Carga el dataset procesado desde la ruta especificada."""
    if path is None:
        path = config.PROCESSED_DATA_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo no se encuentra en: {path}")
        
    df = pd.read_csv(path, encoding="utf-8")
    return df

def info_dataset(df):
    """Muestra información sobre el dataset."""
    print(df.info())