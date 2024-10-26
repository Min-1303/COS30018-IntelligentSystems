from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

def build_model(input_shape):
    """
    Build an LSTM-based neural network model with dropout layers for stock price prediction.
    """
    model = Sequential()

    # Add LSTM layers with dropout to prevent overfitting
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout layer with 20% rate
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Final dense layer with a single unit for output
    model.add(Dense(units=1))

    # Compile the model with Adam optimizer and mean squared error loss function
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    """
    Train the LSTM model on the provided training data.
    """
    # Fit the model to the training data for specified epochs and batch size
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def test_model(model, data, scaler, prediction_days, price_value):
    """
    Use the model to make predictions on test data by preparing the input data,
    scaling it, and transforming model outputs back to original scale.
    """
    # Concatenate all relevant data for input preparation
    total_dataset = pd.concat((data[price_value]), axis=0)
    
    # Select the last part of the dataset for model inputs based on prediction days
    model_inputs = total_dataset[len(total_dataset) - len(data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)  # Scale the inputs

    x_test = []  # List to store test sequences

    # Create sequences for testing from the scaled data
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    # Convert test sequences to numpy array and reshape to match model input
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict prices using the trained model and inverse transform to original scale
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices
