import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils.predict import load_model, make_prediction

model = load_model('models/linear_regression_model.pkl')
with open('utils/label_encoder.pkl', 'rb') as file:
    encoders = pickle.load(file)

st.title("Energy Consumption Prediction")

relativeCompactness = st.text_input("Relative Compactness")
surfaceArea = st.text_input("Surface Area(M)")
roofArea = st.text_input("Roof Area(M)")
overallHeight = st.text_input("Overall Height")

uploaded_file = st.file_uploader("Untuk prediksi masal, upload file", type=["xlsx", "xls"])

data = {
    "X1": relativeCompactness,
    "X2": surfaceArea,
    "X4": roofArea,
    "X5": overallHeight,
}

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

def batch_predictions(data, model):
    predictions = model.predict(data)
    return predictions

# Tombol Prediksi
if st.button("Predict"):
    try:
        if uploaded_file:
            result = batch_predictions(transform_data, model)
        else:
            input_data = transform_data(data, encoders)

            result = make_prediction(input_data, model)
            
            st.write("Hasil Prediksi: \n")
            st.write(f"Heating Load: {result[0]} \n")
            st.write(f"Cooling Load: {result[1]} \n")
    except Exception as e:
        st.error(f"Terjadi Kesalahan: {e}")