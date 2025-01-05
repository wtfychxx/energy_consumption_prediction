import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils.predict import load_model, make_prediction

model = load_model('models/linear_regression_model.pkl')
with open('utils/label_encoder.pkl', 'rb') as file:
    encoders= pickle.load(file)
    # decoded_value = encoders['X1'].inverse_transform([0])  # Ubah 0 ke nilai asli
    print(encoders)