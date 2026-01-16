#!/bin/bash
set -e

# 1. Start MLflow UI
echo "Starting MLflow UI..."
uv run mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///app/mlruns &

# 2. Start API Backend
echo "Starting FastAPI Backend..."
uv run uvicorn src.nps_latam.api:app --host 0.0.0.0 --port 8000 &

# 3. Start Streamlit Frontend
echo "Starting Streamlit Frontend on 7860..."
uv run streamlit run src/nps_latam/app.py --server.port 7860 --server.address 0.0.0.0 &

# Keep container alive by waiting for processes
wait -n
