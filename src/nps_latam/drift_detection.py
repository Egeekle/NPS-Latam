import pandas as pd
import json
import os
import sys
import yaml
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.nps_latam import load_processed_dataset, clean_and_save_dataset

def load_config(config_path="config/drift_config.yaml"):
    """Loads YAML configuration."""
    with open(project_root / config_path, "r") as f:
        return yaml.safe_load(f)

def generate_drift_report(reference_path: str, current_path: str, output_path: str = "reports/drift_report.html", column_config=None):
    """
    Generates a Data Drift report comparing reference data (training) vs current data (new batch).
    """
    # Load Data
    try:
        ref_df = pd.read_csv(reference_path)
        cur_df = pd.read_csv(current_path)
        
        # Determine Column Mapping from config if valid
        col_mapping = ColumnMapping()
        if column_config:
            if "target" in column_config:
                col_mapping.target = column_config["target"]
            if "numerical_features" in column_config:
                # Filter strictly those present in DF
                valid_num = [c for c in column_config["numerical_features"] if c in ref_df.columns]
                col_mapping.numerical_features = valid_num
            if "categorical_features" in column_config:
                valid_cat = [c for c in column_config["categorical_features"] if c in ref_df.columns]
                col_mapping.categorical_features = valid_cat

        # Initialize Report
        report = Report(metrics=[
            DataDriftPreset(), 
            TargetDriftPreset(),
        ])
        
        print(f"Calculating Drift comparing {len(ref_df)} ref vs {len(cur_df)} cur rows...")
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_mapping)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save Report
        report.save_html(output_path)
        print(f"Drift report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating drift report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example Usage integrated with YAML
    try:
        config = load_config().get("drift_detection", {})
        
        # Resolve paths
        data_dir = project_root / "Data"
        ref_path = project_root / config.get("reference_data_path", "Data/reference_data.csv")
        cur_path = project_root / config.get("current_data_path", "Data/current_data.csv")
        output_path = project_root / config.get("report_output_path", "reports/data_drift_report.html")
        
        # Check if files exist, if not create dummy split for demo
        if not ref_path.exists() or not cur_path.exists():
            print("Reference or Current data not found. Creating samples from main dataset for demo...")
            main_file = data_dir / "Satisfaccion_pasajeros_limpio.csv"
            if main_file.exists():
                df = pd.read_csv(main_file)
                split_idx = int(len(df) * 0.7)
                ref_df = df.iloc[:split_idx]
                cur_df = df.iloc[split_idx:]
                
                ref_df.to_csv(ref_path, index=False)
                cur_df.to_csv(cur_path, index=False)
                print(f"Created {ref_path} and {cur_path}")
            else:
                print(f"Main dataset not found at {main_file}. Cannot run demo.")
                sys.exit(1)
        
        col_conf = config.get("column_mapping", {})
        generate_drift_report(str(ref_path), str(cur_path), str(output_path), column_config=col_conf)
        
    except Exception as e:
        print(f"Failed to run drift detection: {e}")
