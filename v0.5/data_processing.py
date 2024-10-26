# File: data_processing.py
# Purpose: This module handles data loading and preprocessing tasks.
# It fetches the stock data from the specified source and prepares it
# for model training by scaling and structuring it appropriately.

import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(company, start_date, end_date, nan_handling='drop', fill_value=0,
              cache_dir='data_cache', use_cache=True):
    """
    Load stock data, handle NaN values, and optionally cache the data locally.
    
    Parameters:
        company (str): Stock ticker symbol of the company.
        start_date (str): Start date for fetching data.
        end_date (str): End date for fetching data.
        nan_handling (str): Method for handling missing data (e.g., 'drop', 'fill').
        fill_value (int): Value to fill NaNs if nan_handling is set to 'fill'.
        cache_dir (str): Directory to save cached data.
        use_cache (bool): If True, load data from cache if available.

    Returns:
        DataFrame: The processed stock data.
    """
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{company}_{start_date}_{end_date}.csv"

    # Check if cached data exists, to avoid re-downloading
    if use_cache and os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded data from cache: {cache_file}")
    else:
        # Download the stock data from Yahoo Finance
        data = yf.download(company, start_date, end_date)

        # Handle missing values based on the specified method
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

        # Save the data to cache for future use
        if use_cache:
            data.to_csv(cache_file)
            print(f"Saved data to cache: {cache_file}")

    return data

def prepare_data(data, feature_columns, prediction_days, split_method='ratio',
                 split_ratio=0.8, split_date=None, random_split=False):
    """
    Prepare, scale, and split stock data for model training.
    
    Parameters:
        data (DataFrame): The stock data.
        feature_columns (list): Columns used as features for training.
        prediction_days (int): Number of days to look back in the sequence.
        split_method (str): Method for data split ('ratio' or 'date').
        split_ratio (float): Training data ratio if split_method is 'ratio'.
        split_date (str): Specific date for splitting data if split_method is 'date'.
        random_split (bool): Whether to randomly split the data.

    Returns:
        Tuple: x_train, y_train, x_test, y_test, and scalers for each feature.
    """
    scalers = {}  # Dictionary to hold scalers for each feature
    scaled_data = {}

    # Scale all feature columns together using a single scaler
    scaler_all_features = MinMaxScaler(feature_range=(0, 1))
    scaled_all_features = scaler_all_features.fit_transform(data[feature_columns])

    # Store the scaler for all features combined
    scalers['all_features'] = scaler_all_features

    # Scale individual features separately and store their scalers
    for feature in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scalers[feature] = scaler

    # Prepare sequences for training and testing
    x_data, y_data = [], []
    for x in range(prediction_days, len(scaled_all_features)):
        x_data.append(scaled_all_features[x - prediction_days:x, :])  # Sequence of past days as input
        y_data.append(scaled_all_features[x, 0])  # Target value (e.g., 'Close' price)

    # Convert lists to NumPy arrays for model compatibility
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Split data into training and testing sets
    if split_method == 'date' and split_date:
        # Split data based on a specific date
        split_index = data.index.get_loc(split_date)
        x_train, x_test = x_data[:split_index], x_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]
    elif split_method == 'ratio':
        # Split data based on a specified ratio
        if random_split:
            # Randomly split data with a fixed random state for reproducibility
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=split_ratio, random_state=42)
        else:
            # Sequentially split data
            split_index = int(len(x_data) * split_ratio)
            x_train, x_test = x_data[:split_index], x_data[split_index:]
            y_train, y_test = y_data[:split_index], y_data[split_index:]
    else:
        raise ValueError("Invalid split method.")

    return x_train, y_train, x_test, y_test, scalers
