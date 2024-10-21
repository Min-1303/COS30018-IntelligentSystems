import numpy as np

def predict_next_day(model, model_inputs, scaler, prediction_days):
    real_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    return prediction