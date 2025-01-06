import joblib
from utils.data_preprocessing import preprocess_input

def load_model(path):
    model = joblib.load(path)
    
    return model

def make_prediction(data, model):
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)

    prediction_formatted = prediction[0]

    prediction_formatted[0] = prediction_formatted[0].round(2)
    prediction_formatted[1] = prediction_formatted[1].round(2)

    return prediction_formatted