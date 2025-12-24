# from celery import Celery
# import pandas as pd
# import io
# import os

# # Configure Redis as the broker to manage task loads [cite: 84]
# celery_app = Celery("tasks", 
#                     broker="redis://redis:6379/0", 
#                     backend="redis://redis:6379/1")

# @celery_app.task(bind=True)
# def run_inference_task(self, csv_content, task_type):
#     # Load input data into a DataFrame
#     df = pd.read_csv(io.StringIO(csv_content))
    
#     # --- LOGIC NOTE ---
#     # Since you submit code without models[cite: 8, 22], ensure your 
#     # loading logic points to a directory that can be mounted as a volume.
    
#     # Placeholder for your trained model prediction:
#     if task_type == 'app':
#         # Logic for identifying 128 applications [cite: 5]
#         df['prediction'] = "Predicted_App_Name" 
#     else:
#         # Logic for identifying 5 attribution types [cite: 6]
#         df['prediction'] = "Predicted_Attribution_Type"

#     # MANDATORY: The result MUST have the 'prediction' column [cite: 9, 21]
#     return df.to_json(orient="split")

from celery import Celery
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
def run_inference_task(self, csv_content, task_type):
    # 1. Load data from string
    df = pd.read_csv(io.StringIO(csv_content))
    
    # 2. Path to models (mounted via volume in docker-compose)
    model_path = f"models/model_{task_type}.pkl"
    feature_path = f"models/{task_type}_features.pkl"
    
    if not os.path.exists(model_path):
        return {"error": "Model file not found. Ensure training is complete."}

    # 3. Load model and feature list
    model = joblib.load(model_path)
    required_features = joblib.load(feature_path)
    
    # 4. Preprocess input data
    X = preprocess_data(df)
    
    # 5. Ensure feature alignment (same columns in same order as training)
    X = X.reindex(columns=required_features, fill_value=0)
    
    # 6. Perform batch inference
    predictions = model.predict(X)
    
    # 7. Add mandatory 'prediction' column
    df['prediction'] = predictions
    
    # Return result as JSON
    return df.to_json(orient="split")