# File: model_operations.py
# Purpose: This module contains functions related to building, training,
# and testing the stock prediction model. It defines the architecture
# of various types of Recurrent Neural Networks (RNNs) including LSTM, GRU,
# SimpleRNN, and their bidirectional variants. It also includes functions to
# train the model on the processed data and test its performance on unseen data.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
import numpy as np
import pandas as pd

def build_model(input_shape, num_layers=3, layer_type='LSTM', layer_size=50, dropout_rate=0.2):
    """
    Constructs a Deep Learning model based on the specified RNN layer type.

    Args:
        input_shape (tuple): Shape of the input data (time_steps, features).
        num_layers (int): Number of RNN layers to include in the model.
        layer_type (str): Type of RNN layer to use (LSTM, GRU, RNN, BiLSTM, BiGRU).
        layer_size (int): Number of units (neurons) in each RNN layer.
        dropout_rate (float): Dropout rate for regularization after each RNN layer.

    Returns:
        model (Sequential): A compiled Keras Sequential model.
    """

    # Initialize a Sequential model
    model = Sequential()

    # Add the first RNN layer based on the specified layer type
    # We start with the first layer, considering whether it should return sequences
    # for the next layer based on the number of layers specified
    if layer_type == 'GRU':
        model.add(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(num_layers > 1)), input_shape=input_shape))
    elif layer_type == 'RNN':
        model.add(SimpleRNN(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'LSTM':
        model.add(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'BiGRU':
        model.add(Bidirectional(GRU(units=layer_size, return_sequences=(num_layers > 1)), input_shape=input_shape))
    else:
        # Raise an error if an unsupported layer type is provided
        raise ValueError(f"Unsupported layer_type: {layer_type}")

    # Apply dropout for regularization
    model.add(Dropout(dropout_rate))

    # Add the remaining RNN layers (if num_layers > 1)
    # Each subsequent layer should consider whether to return sequences to the next layer
    for _ in range(1, num_layers):
        if layer_type == 'GRU':
            model.add(GRU(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'BiLSTM':
            model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1))))
        elif layer_type == 'RNN':
            model.add(SimpleRNN(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'LSTM':
            model.add(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'BiGRU':
            model.add(Bidirectional(GRU(units=layer_size, return_sequences=(_ < num_layers - 1))))

        # Apply dropout after each RNN layer for regularization
        model.add(Dropout(dropout_rate))

    # Add the final output layer, which produces a single value as the prediction
    model.add(Dense(units=1))

    # Compile the model using the Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    """
    Trains the model on the given training data.

    Args:
        model (Sequential): The Keras model to be trained.
        x_train (np.ndarray): The input features for training.
        y_train (np.ndarray): The target values for training.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per gradient update.

    Returns:
        model (Sequential): The trained Keras model.
    """
    # Train the model using the provided training data
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def test_model(model, data, scaler, prediction_days, price_value):
    """
    Tests the model on unseen data and returns the predicted stock prices.

    Args:
        model (Sequential): The trained Keras model.
        data (pd.DataFrame): The dataset containing the stock prices.
        scaler (MinMaxScaler): The scaler used to normalize the data.
        prediction_days (int): Number of days to consider for each prediction.
        price_value (str): The column name in the dataset that contains stock prices.

    Returns:
        predicted_prices (np.ndarray): The model's predicted stock prices.
    """
    # Prepare the data for testing
    total_dataset = pd.concat((data[price_value]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Create the input sequences for testing based on the prediction_days
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Use the model to predict the stock prices
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices
