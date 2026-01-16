import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_save_dataset(df, output_path='airline_satisfaction_transformed_clean.csv'):
    df['target'] = df['Satisfaccion'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    columns_to_drop = ['Genero', 'Tipo_Cliente', 'Tipo_Viaje', 'Clase', 'Satisfaccion', 'Satisfaccion_bin']
    
    existing_cols = [c for c in columns_to_drop if c in df.columns]
    df_cleaned = df.drop(columns=existing_cols, errors='ignore')
    
    return df_cleaned

def split_data(df, target_col='target', train_size=0.6, random_state=42):
    """
    Splits the dataframe into training, validation, and test sets (60/20/20).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=random_state
    )

    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid shapes: X={X_valid.shape}, y={y_valid.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test