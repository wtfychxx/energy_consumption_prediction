import joblib
from utils.data_preprocessing import preprocess_input

def load_model(path):
    model = joblib.load(path)
    
    return model

def make_prediction(data, model):
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    return prediction[0]