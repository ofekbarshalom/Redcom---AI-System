import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
from src.utils import preprocess_data

def test_model(task_type, test_csv_path):
    print(f"\n--- Testing {task_type.upper()} Model ---")
    
    # 1. Check if model exists
    model_path = f"models/model_{task_type}.pkl"
    feature_path = f"models/{task_type}_features.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Train it first!")
        return

    # 2. Load model and features
    model = joblib.load(model_path)
    required_features = joblib.load(feature_path)
    
    # 3. Load test data
    df = pd.read_csv(test_csv_path)
    
    # Identify target column based on task
    target_col = 'label' if task_type == 'app' else 'attribution'
    
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in {test_csv_path}")
        return

    y_true = df[target_col]
    
    # 4. Preprocess features
    X = preprocess_data(df)
    
    # Ensure alignment with training features
    X = X.reindex(columns=required_features, fill_value=0)
    
    # 5. Predict
    print(f"Running inference on {len(df)} samples...")
    y_pred = model.predict(X)
    
    # 6. Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Show report for Attribution (since it has only 5 classes)
    if task_type == 'att':
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    else:
        # For App (128 classes), just show top 10 rows of results
        results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        print("\nFirst 10 Predictions:")
        print(results.head(10))

if __name__ == "__main__":
    # Test App model
    test_model('app', 'data/APP-1/radcom_app_test.csv')
    
    # Test Attribution model
    test_model('att', 'data/attribution/radcom_att_test.csv')