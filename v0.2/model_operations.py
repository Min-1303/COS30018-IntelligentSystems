from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

def build_model(input_shape):
    """
    Build an LSTM-based neural network model with dropout layers for stock prediction.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    """
    Train the model on the provided training data.
    """
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def test_model(model, data, scaler, prediction_days, price_value):
    """
    Use the model to make predictions on test data and return the predicted prices.
    """
    total_dataset = pd.concat((data[price_value]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Prepare test sequences for prediction
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test).reshape(-1, prediction_days, 1)
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices
