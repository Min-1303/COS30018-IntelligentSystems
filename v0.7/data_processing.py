import os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from fredapi import Fred
import numpy as np


# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, column='Close', period=14):
    delta = data[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Function to calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def load_data(company, start_date, end_date, cache_dir='data_cache', use_cache=True):
    """
    Load stock data, handle NaN values, and optionally cache the data locally.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{company}_{start_date}_{end_date}.csv"

    if use_cache and os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded data from cache: {cache_file}")
    else:
        data = yf.download(company, start=start_date, end=end_date)
        data.to_csv(cache_file)
        print(f"Saved data to cache: {cache_file}")

    # Calculate RSI and MACD after loading stock data
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)

    return data


def prepare_data(data, feature_columns, prediction_days, split_method='ratio', split_ratio=0.8, split_date=None):
    """
    Prepare, scale, and split stock data for model training.
    """
    # Add RSI and MACD to the feature columns if needed
    feature_columns += ['RSI', 'MACD']

    # Forward-fill any remaining NaN values
    data = data.ffill().bfill()  # Forward fill and backward fill to handle NaN values

    scalers = {}
    for feature in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scalers[feature] = scaler

    x_data, y_data = [], []
    for x in range(prediction_days, len(data)):
        x_data.append(data[feature_columns].iloc[x - prediction_days:x].values)
        y_data.append(data['Close'].iloc[x])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    if split_method == 'date' and split_date:
        split_index = data.index.get_loc(split_date)
        x_train, x_test = x_data[:split_index], x_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]
    else:
        split_index = int(len(x_data) * split_ratio)
        x_train, x_test = x_data[:split_index], x_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]

    return x_train, y_train, x_test, y_test, scalers



def load_macro_data():
    """
    Load macroeconomic data (e.g., GDP, inflation, unemployment).
    """
    fred = Fred(api_key='233d59032ace03cee4809366959dfa40')

    gdp = fred.get_series('GDP', start_date='2020-01-01', end_date='2023-08-01')
    inflation = fred.get_series('CPIAUCSL', start_date='2020-01-01', end_date='2023-08-01')
    unemployment = fred.get_series('UNRATE', start_date='2020-01-01', end_date='2023-08-01')

    macro_data = pd.DataFrame({
        'GDP': gdp,
        'Inflation': inflation,
        'Unemployment': unemployment
    })

    return macro_data
