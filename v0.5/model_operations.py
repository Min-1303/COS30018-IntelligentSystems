# File: model_operations.py
# Purpose: This module contains functions related to building, training,
# and testing the stock prediction model. It defines various RNN architectures
# (LSTM, GRU, RNN, BiLSTM, BiGRU), trains the model on processed data, 
# and tests its performance on unseen data.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
import numpy as np
import pandas as pd

def build_model(input_shape, num_layers=3, layer_type='LSTM', layer_size=50, dropout_rate=0.2):
    """
    Build a sequential RNN model based on specified architecture and configuration.

    Parameters:
        input_shape (tuple): Shape of input data (time steps, features).
        num_layers (int): Number of RNN layers in the model.
        layer_type (str): Type of RNN layer ('LSTM', 'GRU', 'RNN', 'BiLSTM', 'BiGRU').
        layer_size (int): Number of units in each RNN layer.
        dropout_rate (float): Dropout rate to prevent overfitting.

    Returns:
        model (Sequential): Compiled Keras sequential model.
    """
    model = Sequential()

    # First RNN layer with specified layer type
    if layer_type == 'LSTM':
        model.add(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'GRU':
        model.add(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'RNN':
        model.add(SimpleRNN(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(num_layers > 1)), input_shape=input_shape))
    elif layer_type == 'BiGRU':
        model.add(Bidirectional(GRU(units=layer_size, return_sequences=(num_layers > 1)), input_shape=input_shape))
    else:
        raise ValueError(f"Unsupported layer_type: {layer_type}")

    # Add dropout to prevent overfitting
    model.add(Dropout(dropout_rate))

    # Remaining RNN layers
    for _ in range(1, num_layers):
        if layer_type == 'LSTM':
            model.add(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'GRU':
            model.add(GRU(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'RNN':
            model.add(SimpleRNN(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'BiLSTM':
            model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1))))
        elif layer_type == 'BiGRU':
            model.add(Bidirectional(GRU(units=layer_size, return_sequences=(_ < num_layers - 1))))

        model.add(Dropout(dropout_rate))

    # Output layer for regression
    model.add(Dense(units=1))

    # Compile the model with mean squared error loss for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    """
    Train the model on the training dataset.

    Parameters:
        model (Sequential): The model to be trained.
        x_train (array): Training data input sequences.
        y_train (array): Training data targets.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        model (Sequential): Trained model.
    """
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def test_model(model, data, scaler, prediction_days, price_value):
    """
    Test the model on unseen data by generating predictions.

    Parameters:
        model (Sequential): Trained model for prediction.
        data (DataFrame): Complete dataset for prediction context.
        scaler (MinMaxScaler): Scaler used for data normalization.
        prediction_days (int): Number of days to look back in the sequence.
        price_value (str): Column name representing the price (e.g., 'Close').

    Returns:
        array: Inverse-scaled predicted prices for the test data.
    """
    # Combine dataset to extract relevant sequence for predictions
    total_dataset = pd.concat((data[price_value]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)  # Normalize input data

    # Generate sequences for testing
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    # Reshape test data to match model input
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Generate predictions and inverse-transform them back to original scale
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices
