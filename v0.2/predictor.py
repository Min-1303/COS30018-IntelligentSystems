import numpy as np

def predict_next_day(model, last_sequence, scaler, prediction_days):
    """
    Predict the stock price for the next day based on the last sequence of data.
    """
    # Reshape the last sequence for prediction
    real_data = last_sequence[-prediction_days:]
    prediction = model.predict(real_data)

    return scaler.inverse_transform(prediction)
