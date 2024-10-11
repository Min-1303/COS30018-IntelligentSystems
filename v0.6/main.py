#main.py

import openpyxl
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from config import CONFIG  # Import the configuration settings

from data_processing import load_data, prepare_data
from model_operations import (
    create_model, train_model, train_arima, predict_arima,
    train_sarima, predict_sarima, train_random_forest, test_random_forest
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Select the configuration set you want to use
CONFIG_SET = '3'  # Change this to switch configs
config = CONFIG[CONFIG_SET]

# Stock symbol for analysis
COMPANY = 'CBA.AX'

# Training and testing date range
TRAIN_START_DATE, TRAIN_END_DATE = '2020-01-01', '2023-08-01'
TEST_START_DATE, TEST_END_DATE = '2023-08-02', '2024-07-02'

# Days to consider for prediction
PREDICTION_DAYS = 60

# Features for training
FEATURES_COLUMNS = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]

# NaN handling method and fill value
NA_METHOD, NA_FILL_VALUE = 'ffill', 0

# Data split method
SPLIT_METHOD = 'ratio'
TRAIN_TEST_RATIO = 0.8
DATE_SPLIT = '2023-01-01'

# Random split flag
USE_RANDOM_SPLIT = False

# Cache options
ENABLE_CACHE = True
CACHE_DIRECTORY = 'data_cache'

# Load and prepare data
dataset = load_data(COMPANY, TRAIN_START_DATE, TRAIN_END_DATE, nan_handling=NA_METHOD, fill_value=NA_FILL_VALUE,
                    cache_dir=CACHE_DIRECTORY, use_cache=ENABLE_CACHE)

# Ensure the date index is in datetime format
dataset.index = pd.to_datetime(dataset.index)

# Reindex the dataset to fill missing dates
date_range = pd.date_range(start=dataset.index.min(), end=dataset.index.max(), freq='D')
dataset = dataset.reindex(date_range).interpolate(method='linear')  # Interpolation for filling values

# Prepare the data
x_train_data, y_train_data, x_test_data, y_test_data, data_scalers = prepare_data(
    dataset, 
    FEATURES_COLUMNS, 
    PREDICTION_DAYS,
    split_method=SPLIT_METHOD,
    split_ratio=TRAIN_TEST_RATIO,
    split_date=DATE_SPLIT,
    random_split=USE_RANDOM_SPLIT
)

# Train ARIMA and SARIMA models
arima_model = train_arima(dataset['Close'], order=config['arima_order'])
sarima_model = train_sarima(dataset['Close'], order=config['sarima_order'], seasonal_order=config['sarima_seasonal_order'])

# Predictions from ARIMA and SARIMA
arima_preds = predict_arima(arima_model, len(y_test_data))
sarima_preds = predict_sarima(sarima_model, len(y_test_data))

input_shape = (x_train_data.shape[1], x_train_data.shape[2])  

# Build and train the LSTM/GRU model
lstm_gru_model = create_model(input_shape=input_shape, config=config)  # Use 'config' here
train_model(lstm_gru_model, x_train_data, y_train_data, config=config)  # Use 'config' here

# Predictions from LSTM/GRU model
predicted_prices = lstm_gru_model.predict(x_test_data)
predicted_prices = data_scalers["Close"].inverse_transform(predicted_prices)

# Inverse transform the y_test data
y_test_unscaled = data_scalers["Close"].inverse_transform(y_test_data.reshape(-1, 1))

# Flatten predictions
predicted_prices_flattened = predicted_prices.flatten()
arima_preds_flat = arima_preds.values.flatten()
sarima_preds_flat = sarima_preds.values.flatten()

# Calculate model errors to determine weights
arima_error = np.mean((arima_preds_flat - y_test_unscaled.flatten())**2)
sarima_error = np.mean((sarima_preds_flat - y_test_unscaled.flatten())**2)
lstm_error = np.mean((predicted_prices_flattened - y_test_unscaled.flatten())**2)
total_error = arima_error + sarima_error + lstm_error

# Assign weights inversely proportional to MSE
arima_weight = (1 - arima_error / total_error)
sarima_weight = (1 - sarima_error / total_error)
lstm_weight = (1 - lstm_error / total_error)

# Normalize weights
total_weight = arima_weight + sarima_weight + lstm_weight
arima_weight /= total_weight
sarima_weight /= total_weight
lstm_weight /= total_weight

# Combine predictions
combined_predictions = (arima_weight * arima_preds_flat) + (sarima_weight * sarima_preds_flat) + (lstm_weight * predicted_prices_flattened)

# Train Random Forest Model
rf_model = train_random_forest(
    x_train_data.reshape(x_train_data.shape[0], -1), 
    y_train_data,
    n_estimators=config['rf_n_estimators'],
    max_depth=config['rf_max_depth'],
    min_samples_split=config['rf_min_samples_split']
)
rf_predictions = test_random_forest(rf_model, x_test_data.reshape(x_test_data.shape[0], -1))

# Inverse transform Random Forest predictions
rf_predictions = data_scalers["Close"].inverse_transform(rf_predictions.reshape(-1, 1))

# Visualization of Predictions
plt.figure(figsize=(14, 8))
plt.plot(y_test_unscaled, color="black", linewidth=2, label=f"Actual {COMPANY} Price")
plt.plot(combined_predictions, color="green", linewidth=2, linestyle='--', label="Combined Prediction")
plt.plot(predicted_prices_flattened, color="red", linestyle='--', label="LSTM/GRU Prediction")
plt.plot(arima_preds_flat, color="yellow", linestyle=':', label="ARIMA Prediction")
plt.plot(sarima_preds_flat, color="blue", linestyle='-.', label="SARIMA Prediction")
plt.plot(rf_predictions, color="pink", linestyle='-', label="Random Forest Prediction")

plt.title(f"{COMPANY} Share Price Prediction", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel(f"{COMPANY} Price", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# Prepare final data for Excel
final_data = {
    'Date': dataset.index[-len(y_test_data):],
    'Actual Price': y_test_unscaled.flatten(),
    'ARIMA Prediction': arima_preds_flat,
    'SARIMA Prediction': sarima_preds_flat,
    'LSTM/GRU Prediction': predicted_prices_flattened,
    'Combined Prediction': combined_predictions,
    'Random Forest Prediction': rf_predictions.flatten()
}

# Create a DataFrame for final data
final_df = pd.DataFrame(final_data)

# Specify the output file path
output_file_path = 'final_predictions_3.xlsx'

# Save the DataFrame to an Excel file
final_df.to_excel(output_file_path, index=False)

print(f"Predictions have been saved to {output_file_path}")