import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(company, start_date, end_date, nan_handling='drop', fill_value=0,
              cache_dir='data_cache', use_cache=True):
    """
    Load stock data for the specified company and date range, handle NaN values,
    and cache the data locally if enabled.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{company}_{start_date}_{end_date}.csv"

    # Check if cached data exists, else download and cache it
    if use_cache and os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded data from cache: {cache_file}")
    else:
        data = yf.download(company, start_date, end_date)

        # Handle NaN values based on the specified method
        if nan_handling == 'drop':
            data.dropna(inplace=True)
        elif nan_handling == 'fill':
            data.fillna(fill_value, inplace=True)
        elif nan_handling == 'ffill':
            data.ffill(inplace=True)
        elif nan_handling == 'bfill':
            data.bfill(inplace=True)
        else:
            raise ValueError("Invalid NaN handling method.")

        # Cache the data if required
        if use_cache:
            data.to_csv(cache_file)
            print(f"Saved data to cache: {cache_file}")

    return data

def prepare_data(data, feature_columns, prediction_days, split_method='ratio',
                 split_ratio=0.8, split_date=None, random_split=False):
    """
    Scale and split the data into training and testing sets, using the specified features
    and prediction days for model input preparation.
    """
    scalers = {}
    scaled_data = {}

    # Scale each feature column and store the corresponding scaler
    for feature in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scalers[feature] = scaler

    x_data, y_data = [], []
    # Prepare input sequences (x_data) and corresponding outputs (y_data)
    for x in range(prediction_days, len(scaled_data[feature_columns[0]])):
        x_data.append(np.hstack([scaled_data[feature][x - prediction_days:x, 0] for feature in feature_columns]))
        y_data.append(scaled_data[feature_columns[0]][x, 0])  # Assuming 'Close' or first feature column for y_data

    x_data = np.array(x_data).reshape(-1, prediction_days, len(feature_columns))
    y_data = np.array(y_data)

    # Split data into training and testing sets based on specified method
    if split_method == 'date' and split_date:
        split_index = data.index.get_loc(split_date)
        x_train, x_test = x_data[:split_index], x_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]
    elif split_method == 'ratio':
        if random_split:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=split_ratio, random_state=42)
        else:
            split_index = int(len(x_data) * split_ratio)
            x_train, x_test = x_data[:split_index], x_data[split_index:]
            y_train, y_test = y_data[:split_index], y_data[split_index:]
    else:
        raise ValueError("Invalid split method.")

    return x_train, y_train, x_test, y_test, scalers
