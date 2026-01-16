from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
import sys
import os
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.nps_latam import (
    load_processed_dataset, 
    clean_and_save_dataset, 
    split_data, 
    create_logreg_pipeline,
    FlightChatbot
)
from src.nps_latam.genai_features import analyze_feedback_batch

app = FastAPI(title="NPS Latam API", description="API for Flight Satisfaction Prediction and Chatbot", version="1.0.0")

# --- Global State ---
model_pipeline = None
model_features = None
chatbot_instance = None

# --- Pydantic Data Models ---
class PassengerFeatures(BaseModel):
    # Dictionary to allow dynamic features matching the training set
    # Expected keys: 'Gender', 'age', 'Wif_a_bordo', etc. (as per processed dataset columns)
    data: dict

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    texts: List[str]

# --- Startup Event ---
@app.on_event("startup")
def startup():
    global model_pipeline, model_features, chatbot_instance
    
    print("Starting NPS Latam API Services...")
    
    # 1. Initialize Chatbot
    try:
        log_path = project_root / "Data" / "chatbot_logs.csv"
        chatbot_instance = FlightChatbot(log_file=str(log_path))
        print("✅ Chatbot initialized.")
    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        
    # 2. Train and Load Model
    # In a production environment, we would load a serialized .pkl file.
    # Here, we train largely for demonstration purposes on startup.
    try:
        print("Training model on current data...")
        df_raw = load_processed_dataset()
        
        # Ensure 'target' exists (it might not be in the raw load_processed_dataset if it wasn't cleaned/saved back properly)
        # We run the cleaning pipeline just to be safe and ensure consistent state
        df_processed = clean_and_save_dataset(df_raw, output_path=str(project_root / "Data" / "temp_api_training.csv"))
        
        # Split
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df_processed)
        
        # Store feature names to align input later
        model_features = list(X_train.columns)
        
        # Create and Fit Pipeline
        model_pipeline = create_logreg_pipeline()
        model_pipeline.fit(X_train, y_train)
        print(f"✅ Model trained on {len(X_train)} records. Features: {len(model_features)}")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_pipeline is not None, "chatbot_loaded": chatbot_instance is not None}

@app.post("/predict")
def predict(features: PassengerFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([features.data])
        
        # Align columns with training data
        # Fill missing with 0, drop extras, strictly order columns
        input_df_aligned = input_df.reindex(columns=model_features, fill_value=0)
        
        # Predict
        prediction = model_pipeline.predict(input_df_aligned)[0]
        prob = model_pipeline.predict_proba(input_df_aligned)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(prob),
            "label": "Satisfied" if prediction == 1 else "Neutral/Dissatisfied"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if chatbot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot is not available.")
    
    try:
        response = chatbot_instance.respond(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/analyze_feedback")
def analyze_endpoint(request: FeedbackRequest):
    try:
        # This uses the external GenAI module
        if not request.texts:
            return []
            
        df_result = analyze_feedback_batch(request.texts)
        # Convert to list of dicts
        return df_result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
