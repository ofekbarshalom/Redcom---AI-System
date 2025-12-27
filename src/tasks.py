from celery import Celery
from typing import Any
import pandas as pd
import joblib
import io
import os
from src.utils import preprocess_data

# Initialize Celery with Redis broker
celery_app = Celery("tasks", 
                    broker="redis://redis:6379/0", 
                    backend="redis://redis:6379/1")

@celery_app.task(bind=True)
def run_inference_task(self, csv_content: str, task_type: str) -> str | dict[str, str]:
    # Load data from string
    df = pd.read_csv(io.StringIO(csv_content))
    
    model_path = f"models/model_{task_type}.pkl"
    feature_path = f"models/{task_type}_features.pkl"
    
    if not os.path.exists(model_path):
        return {"error": "Model file not found. Ensure training is complete."}

    # Load model and feature list
    model = joblib.load(model_path)
    required_features = joblib.load(feature_path)
    
    X = preprocess_data(df)
    
    # Ensure feature alignment (same columns in same order as training)
    X = X.reindex(columns=required_features, fill_value=0)
    
    # Perform batch inference
    predictions = model.predict(X)
    
    df['prediction'] = predictions
    
    # Return result as JSON
    return df.to_json(orient="split")