import streamlit as st
import pandas as pd
import pickle
from utils.predict import load_model, make_prediction
from utils.data_transformers import transform_data, upload_transform_data

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

def batch_predictions(data, model):
    predictions = model.predict(data)
    return predictions

if uploaded_file:
    #membaca file excel
    input_data = pd.read_excel(uploaded_file)

    # Tampilkan 5 data teratas
    st.write("5 Data Teratas:")
    st.dataframe(input_data.head())

# Tombol Prediksi
if st.button("Predict"):
    try:
        if uploaded_file:
            # transformed_data = upload_transform_data(input_data, encoders)

            predictions = batch_predictions(input_data, model)

            input_data['Y1'] = predictions[:, 0]
            input_data['Y2'] = predictions[:, 1]
            
            # menambahkan kolom ke dalam input_data menggunakan hasil predictions

            st.write("Hasil Prediksi: ")
            st.dataframe(input_data.head())

            # Simpan hasil ke file Excel
            output_file = "hasil_prediksi.xlsx"
            input_data.to_excel(output_file, index=False)

            # Tombol download
            with open(output_file, "rb") as file:
                st.download_button(
                    label="Download Hasil Prediksi",
                    data=file,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            input_data = transform_data(data, encoders)

            result = make_prediction(input_data, model)
            
            st.write("Hasil Prediksi: \n")
            st.write(f"Heating Load: {result[0]} \n")
            st.write(f"Cooling Load: {result[1]} \n")
    except Exception as e:
        st.error(f"Terjadi Kesalahan saat memproses data: {e}")