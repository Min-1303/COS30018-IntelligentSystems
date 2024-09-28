import statsmodels.api as sm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from sklearn.ensemble import RandomForestRegressor
from config import CONFIG  # Import the config



def create_model(input_shape, config):
    """
    Constructs a Deep Learning model with specified RNN types.
    """
    model = Sequential()

     # Read RNN type and bidirectionality from config
    rnn_type = config['rnn_type']  
    use_bidirectional = config['use_bidirectional']  
    # Get LSTM/GRU configuration from the config
    layers = config['lstm_layers'] if rnn_type == 'LSTM' else config['gru_layers']
    units = config['lstm_units'] if rnn_type == 'LSTM' else config['gru_units']
    dropout = config['lstm_dropout'] if rnn_type == 'LSTM' else config['gru_dropout']

    # Add the first RNN layer
    if use_bidirectional:
        if rnn_type == 'LSTM':
            model.add(Bidirectional(LSTM(units=units, return_sequences=(layers > 1), input_shape=input_shape)))
        elif rnn_type == 'GRU':
            model.add(Bidirectional(GRU(units=units, return_sequences=(layers > 1), input_shape=input_shape)))
    else:
        if rnn_type == 'LSTM':
            model.add(LSTM(units=units, return_sequences=(layers > 1), input_shape=input_shape))
        elif rnn_type == 'GRU':
            model.add(GRU(units=units, return_sequences=(layers > 1), input_shape=input_shape))

    model.add(Dropout(dropout))

    # Add remaining RNN layers
    for _ in range(1, layers):
        if rnn_type == 'LSTM':
            model.add(LSTM(units=units, return_sequences=(_ < layers - 1)))
        elif rnn_type == 'GRU':
            model.add(GRU(units=units, return_sequences=(_ < layers - 1)))
        model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, train_x, train_y, config):
    """
    Trains the given model with specified training data.
    """
    model.fit(train_x, train_y, epochs=config['epochs'], batch_size=config['batch_size'])
    return model

def train_arima(data, order):
    model = sm.tsa.ARIMA(data, order=order)
    return model.fit()

def train_sarima(data, order, seasonal_order):
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order)
    return model.fit()

def predict_arima(model, steps):
    return model.forecast(steps=steps)

def predict_sarima(model, steps):
    return model.forecast(steps=steps)

def train_random_forest(train_x, train_y, n_estimators=100, max_depth=None, min_samples_split=2):
    """
    Trains a Random Forest model with the specified parameters.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    rf_model.fit(train_x, train_y)
    return rf_model

def test_random_forest(model, test_x):
    return model.predict(test_x)
