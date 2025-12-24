import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import sys

# Add src directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import preprocess_data

def train_app_model(csv_path):
    print(f"Loading App training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Separate the target label
    y = df['label']
    # Preprocess features
    X = preprocess_data(df)
    
    print(f"Training App Model (Random Forest)... Features: {X.shape[1]}")
    # n_jobs=-1 uses all available CPU cores for faster training
    model = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    model.fit(X, y)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model and the feature list to ensure alignment during inference
    joblib.dump(model, 'models/model_app.pkl')
    joblib.dump(X.columns.tolist(), 'models/app_features.pkl')
    print("App Model saved successfully to models/model_app.pkl")

if __name__ == "__main__":
    train_app_model('data/APP-1/radcom_app_train.csv')