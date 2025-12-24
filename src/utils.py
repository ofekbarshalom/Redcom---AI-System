# import pandas as pd
# import numpy as np

# def preprocess_data(df: pd.DataFrame):
#     """
#     Cleans and prepares the 111 features for the model[cite: 14].
#     """
#     # 1. Drop the label/attribution columns if they exist in the input
#     cols_to_drop = ['label', 'attribution', 'prediction']
#     df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

#     # 2. Basic Feature Engineering: Convert Protocol to numeric if it's 'tcp'/'udp'
#     if 'Protocol' in df.columns:
#         df['Protocol'] = df['Protocol'].map({'tcp': 6, 'udp': 17}).fillna(0)

#     # 3. Handle IP Addresses: Radcom data often requires numerical formats [cite: 14]
#     # Simple approach: Convert IP strings to integer representation
#     for col in ['Source_IP', 'Destination_IP']:
#         if col in df.columns:
#             df[col] = df[col].apply(lambda x: int(''.join([f"{int(i):08b}" for i in x.split('.')]), 2) if isinstance(x, str) else 0)

#     # 4. Fill missing values to prevent inference crashes
#     df = df.fillna(0)

#     return df

# def format_output(original_df: pd.DataFrame, predictions: np.ndarray):
#     """
#     Adds the mandatory 'prediction' column to the original data[cite: 9].
#     """
#     output_df = original_df.copy()
#     output_df['prediction'] = predictions
#     return output_df


import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame):
    """
    Cleans and prepares the 111 traffic features for the model.
    Ensures that both training and inference data follow the same logic.
    """
    df = df.copy()
    
    # 1. Convert IP Addresses to 32-bit integers
    for col in ['Source_IP', 'Destination_IP']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: sum([int(b) << 8*(3-i) for i, b in enumerate(x.split('.'))]) if isinstance(x, str) else 0)
    
    # 2. Map Protocol strings to numeric values
    if 'Protocol' in df.columns:
        protocol_map = {'tcp': 6, 'udp': 17}
        df['Protocol'] = df['Protocol'].map(protocol_map).fillna(0)
    
    # 3. Drop target labels and non-numeric features like Timestamp
    # These should not be used as features for prediction
    cols_to_drop = ['label', 'attribution', 'prediction', 'Timestamp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 4. Fill missing values (NaN) with 0 to prevent crashes
    df = df.fillna(0)
    
    return df