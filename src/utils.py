import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame):
    """
    Cleans the 111 features by removing identity-based columns 
    and focusing on behavioral/statistical patterns.
    """
    df = df.copy()
    
    # 1. DROP IDENTITY & IRRELEVANT FEATURES
    # We remove IPs, Ports, and Timestamps to avoid Overfitting.
    drop_cols = [
        'Source_IP', 'Destination_IP', 'Source_port', 'Destination_port', 
        'Timestamp', 'label', 'attribution', 'prediction'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 2. ENCODE PROTOCOL
    # Protocol is important for Attribution (e.g., Video often uses UDP).
    if 'Protocol' in df.columns:
        df['Protocol'] = df['Protocol'].map({'tcp': 6, 'udp': 17}).fillna(0)
        
    # 3. FILL MISSING VALUES
    # Ensure all numerical data is ready for the Random Forest model.
    df = df.fillna(0)
    
    return df