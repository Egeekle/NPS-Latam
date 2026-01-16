import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os
import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.nps_latam.data_pipeline import clean_and_save_dataset, split_data
from src.nps_latam.genai_features import analyze_feedback_batch

def train_and_track():
    mlflow.set_tracking_uri("file://" + str(project_root / "mlruns"))
    mlflow.set_experiment("NPS_Latam_Model_Tracking")
    
    with mlflow.start_run():
        print("Starting MLflow run...")
        
        # 1. Load Data
        data_path = project_root / "Data" / "Satisfaccion_pasajeros_limpio.csv"
        if not data_path.exists():
            print(f"Data not found at {data_path}")
            return
            
        df = pd.read_csv(data_path)
        
        # 2. Integrate Chatbot Logs (as requested)
        # We attempt to load logs to potential use as additional data or just log stats
        logs_path = project_root / "Data" / "chatbot_logs.csv"
        if logs_path.exists():
            logs_df = pd.read_csv(logs_path)
            mlflow.log_metric("chatbot_interactions_count", len(logs_df))
            print(f"Found {len(logs_df)} chatbot interactions.")
            
            # Feature Idea: Use GenAI to extract sentiment from logs and maybe perform online learning
            # For now, we just track the volume.
        
        # 3. Preprocess
        # Using existing pipeline
        try:
             # Ensure target mapping explicitly if needed, or rely on clean_and_save_dataset
            df_clean = clean_and_save_dataset(df, output_path=None)
            X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df_clean)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return

        # 4. Model Training
        n_estimators = 100
        max_depth = 10
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        # 5. Evaluation
        y_pred = clf.predict(X_valid)
        y_prob = clf.predict_proba(X_valid)[:, 1]
        
        acc = accuracy_score(y_valid, y_pred)
        f1 = f1_score(y_valid, y_pred)
        auc = roc_auc_score(y_valid, y_prob)
        
        print(f"Validation Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # 6. Log Model
        mlflow.sklearn.log_model(clf, "random_forest_model")
        
        # 7. Feature Importance Plot
        feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_imp.head(20), y=feature_imp.head(20).index)
        plt.title("Top 20 Feature Importance")
        plt.tight_layout()
        plot_path = project_root / "reports" / "feature_importance.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        mlflow.log_artifact(str(plot_path))
        print("Run complete.")

if __name__ == "__main__":
    train_and_track()
