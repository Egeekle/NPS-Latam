import pandas as pd 
import os 





def clean_and_save_dataset(df, output_path='airline_satisfaction_transformed_clean.csv'):
    columns_to_drop = ['Genero', 'Tipo_Cliente', 'Tipo_Viaje', 'Clase', 'Satisfaccion', 'Satisfaccion_bin']
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    df_cleaned.to_csv(output_path, index=False)
    print(f"Columnas eliminadas: {columns_to_drop}\nDataFrame limpio guardado en: {output_path}")

    return df_cleaned

