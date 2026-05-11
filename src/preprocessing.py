import pandas as pd
import numpy as np

def preprocess_data(df):
    """Handle missing values and encode categorical columns."""
    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Encode binary columns
    if 'daynight' in df.columns:
        df['daynight'] = df['daynight'].map({'D': 1, 'N': 0})
        
    if 'satellite' in df.columns:
        df['satellite'] = df['satellite'].map({'Terra': 0, 'Aqua': 1})
        
    return df
