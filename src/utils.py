import pandas as pd

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    
    # We remove IPs, Ports, and Timestamps to avoid Overfitting.
    drop_cols = [
        'Source_IP', 'Destination_IP', 'Source_port', 'Destination_port', 
        'Timestamp', 'label', 'attribution', 'prediction'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Encode Protocol
    if 'Protocol' in df.columns:
        df['Protocol'] = df['Protocol'].map({'tcp': 6, 'udp': 17}).fillna(0)
        
    # FIll Missing Values
    df = df.fillna(0)
    
    return df