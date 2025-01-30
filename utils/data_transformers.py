import numpy as np
import pandas as pd

def upload_transform_data(data, encoders):
    """
    Transform data using the provided encoders and handle missing values.
    
    Parameters:
    - data: DataFrame, the input data to process.
    - encoders: dict, a dictionary where keys are column names, and values are encoders.
    
    Returns:
    - DataFrame, transformed and cleaned data.
    """
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    # Handle columns with encoders
    for column, encoder in encoders.items():
        if column in data.columns:
            # Fill missing values with "Unknown"
            data[column] = data[column].fillna("Unknown")
            
            # Transform values using the encoder
            data[column] = data[column].map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else np.nan
            )
    
    # Handle columns without encoders
    for column in data.columns:
        if column not in encoders:
            if pd.api.types.is_object_dtype(data[column]):  # For categorical columns
                data[column] = data[column].fillna("Unknown")
            else:  # For numeric columns
                data[column] = data[column].fillna(data[column].mean())
    
    # Validate and handle any remaining NaN values
    for column in data.columns:
        if data[column].isna().any():
            if pd.api.types.is_object_dtype(data[column]):  # For categorical columns
                data[column] = data[column].fillna("Unknown")
            else:  # For numeric columns
                data[column] = data[column].fillna(data[column].mean())
    
    return data


def transform_data(data, encoders):
    transformed_data = {}
    for key, value in data.items():
        if key in encoders:  # Jika kolom memiliki encoder
            if value not in encoders[key].classes_:
                # Tambahkan nilai baru sementara ke classes_
                encoders[key].classes_ = np.append(encoders[key].classes_, value)
            
                # Transformasikan nilai ke bentuk numerik
                transformed_data[key] = encoders[key].transform([value])[0]
        else:
            # Jika tidak ada encoder, gunakan nilai asli
            transformed_data[key] = value
    return pd.DataFrame([transformed_data])