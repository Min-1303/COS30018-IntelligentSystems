# File: data_processing.py
# Purpose: This module handles data loading and preprocessing tasks.
# It fetches the stock data from the specified source and prepares it
# for model training by scaling and structuring it appropriately.

import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(company, start_date, end_date, nan_handling='drop', fill_value=0,
              cache_dir='data_cache', use_cache=True):
    """
    Load stock data from Yahoo Finance, handling NaN values and optionally caching the data locally.

    Parameters:
        company (str): Stock ticker symbol of the company.
        start_date (str): Start date for fetching data (YYYY-MM-DD).
        end_date (str): End date for fetching data (YYYY-MM-DD).
        nan_handling (str): Method for handling NaN values ('drop', 'fill', 'ffill', 'bfill').
        fill_value (float): Value to fill NaNs if 'fill' is chosen.
        cache_dir (str): Directory for caching downloaded data.
        use_cache (bool): Flag to use cached data if available.

    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{company}_{start_date}_{end_date}.csv"

    # Check if cached data exists
    if use_cache and os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded data from cache: {cache_file}")
    else:
        # Download data from Yahoo Finance
        data = yf.download(company, start=start_date, end=end_date)

        # Handle NaN values based on specified method
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

        # Save data to cache if not using cache
        if use_cache:
            data.to_csv(cache_file)
            print(f"Saved data to cache: {cache_file}")

    return data


def prepare_data(data, feature_columns, prediction_days, split_method='ratio',
                 split_ratio=0.8, split_date=None, random_split=False):
    """
    Prepare, scale, and split stock data for model training.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data.
        feature_columns (list): List of feature columns to be used for training.
        prediction_days (int): Number of days to use for prediction.
        split_method (str): Method for splitting data ('date' or 'ratio').
        split_ratio (float): Ratio for training/testing split (only used if split_method is 'ratio').
        split_date (str): Date to split data (only used if split_method is 'date').
        random_split (bool): If True, uses random splitting instead of sequential splitting.

    Returns:
        tuple: x_train, y_train, x_test, y_test, scalers (train/test data and scalers).
    """
    scalers = {}
    scaled_data = {}

    # Create a scaler for all features combined
    scaler_all_features = MinMaxScaler(feature_range=(0, 1))
    scaled_all_features = scaler_all_features.fit_transform(data[feature_columns])
    scalers['all_features'] = scaler_all_features

    # Scale individual features and store their scalers
    for feature in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scalers[feature] = scaler

    x_data, y_data = [], []
    # Prepare data for supervised learning
    for x in range(prediction_days, len(scaled_all_features)):
        x_data.append(scaled_all_features[x - prediction_days:x, :])
        y_data.append(scaled_all_features[x, 0])  # Assuming 'Close' or the first column is the target

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Split data based on specified method
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
