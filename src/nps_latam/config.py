import os
import sys
from pathlib import Path

# Resolve the project root relative to this file
# config.py is in src/nps_latam/ -> parent is src/ -> parent is Project Root
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent

# Add Project Root to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    print(f"Project Root added to sys.path: {PROJECT_ROOT}")

# Define Data Paths
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DATA_PATH = DATA_DIR / "Satisfaccion_pasajeros_limpio.csv"
