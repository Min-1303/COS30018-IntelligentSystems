import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR

# XGBoost Model Functions
def train_xgboost_model(x_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    Train the XGBoost model.
    """
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    xgb_model.fit(x_train_reshaped, y_train)
    return xgb_model

def test_xgboost_model(xgb_model, x_test):
    """
    Test the XGBoost model.
    """
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
    return xgb_model.predict(x_test_reshaped)


# SVR Model Functions
def train_svr_model(x_train, y_train, kernel='rbf', C=100, gamma=0.1):
    """
    Train the SVR model.
    """
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    svr_model = SVR(kernel=kernel, C=C, gamma=gamma)
    svr_model.fit(x_train_reshaped, y_train)
    return svr_model

def test_svr_model(svr_model, x_test):
    """
    Test the SVR model.
    """
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
    return svr_model.predict(x_test_reshaped)


# Random Forest Model Functions
def train_random_forest_model(x_train, y_train, n_estimators=100, max_depth=10, min_samples_split=5):
    """
    Train the Random Forest model.
    """
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    rf_model.fit(x_train_reshaped, y_train)
    return rf_model

def test_random_forest_model(rf_model, x_test):
    """
    Test the Random Forest model.
    """
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
    return rf_model.predict(x_test_reshaped)


# Train and Predict with XGBoost, SVR, and Random Forest
def train_and_predict_models(x_train, y_train, x_test, y_test, scalers):
    """
    Train and predict using XGBoost, SVR, and Random Forest models.
    """
    # Train XGBoost
    xgb_model = train_xgboost_model(x_train, y_train)
    xgb_predictions = test_xgboost_model(xgb_model, x_test)

    # Train SVR
    svr_model = train_svr_model(x_train, y_train)
    svr_predictions = test_svr_model(svr_model, x_test)

    # Train Random Forest
    rf_model = train_random_forest_model(x_train, y_train)
    rf_predictions = test_random_forest_model(rf_model, x_test)

    # Inverse transform the predictions to match the original scale
    xgb_predictions = scalers['Close'].inverse_transform(xgb_predictions.reshape(-1, 1))
    svr_predictions = scalers['Close'].inverse_transform(svr_predictions.reshape(-1, 1))
    rf_predictions = scalers['Close'].inverse_transform(rf_predictions.reshape(-1, 1))
    y_test_unscaled = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

    return xgb_predictions, svr_predictions, rf_predictions, y_test_unscaled


# Plot predictions (e.g., XGBoost, SVR, Random Forest)
def plot_predictions(y_test_unscaled, xgb_predictions, svr_predictions, rf_predictions):
    plt.figure(figsize=(8, 4))
    plt.plot(y_test_unscaled, label='Actual Prices', color='blue', linewidth=1.5)
    plt.plot(xgb_predictions, label='XGBoost Predictions', color='green', linestyle='--', linewidth=1.5)
    plt.plot(svr_predictions, label='SVR Predictions', color='red', linestyle='--', linewidth=1.5)
    plt.plot(rf_predictions, label='Random Forest Predictions', color='orange', linestyle='--', linewidth=1.5)
    plt.title('Stock Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Get the current figure manager and set position using wm_geometry for Tkinter
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+0+0")  # Position it at the top-left corner of the screen

    plt.show(block=False)

# Plot RSI and MACD indicators
def plot_rsi_macd(data):
    plt.figure(figsize=(8, 4))

    # Plot RSI
    plt.subplot(2, 1, 1)
    plt.plot(data['RSI'], label='RSI', color='blue', linewidth=1.5)
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Plot MACD
    plt.subplot(2, 1, 2)
    plt.plot(data['MACD'], label='MACD', color='blue', linewidth=1.5)
    plt.plot(data['MACD_Signal'], label='MACD Signal', color='red', linestyle='--', linewidth=1.5)
    plt.title('Moving Average Convergence Divergence (MACD)')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()

    # Set window position using wm_geometry for Tkinter
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+0+650")  # Position it at the bottom-left corner of the screen
    plt.show(block=False)

# Plot Additional Indicators (Bollinger Bands, Stochastic Oscillator, ADX, CCI)
def plot_additional_indicators(data):
    plt.figure(figsize=(10, 8))  # Increased figure size to accommodate all plots

    # Plot Bollinger Bands
    plt.subplot(4, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue', linewidth=1)
    plt.plot(data['Bollinger_Middle'], label='Bollinger Middle Band', color='orange', linestyle='--')
    plt.plot(data['Bollinger_Upper'], label='Bollinger Upper Band', color='green', linestyle='--')
    plt.plot(data['Bollinger_Lower'], label='Bollinger Lower Band', color='red', linestyle='--')
    plt.title('Bollinger Bands with Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

    # Plot Stochastic Oscillator
    plt.subplot(4, 1, 2)
    plt.plot(data['%K'], label='Stochastic %K', color='purple', linewidth=1.2)
    plt.plot(data['%D'], label='Stochastic %D (Signal)', color='orange', linestyle='--')
    plt.title('Stochastic Oscillator (%K and %D)')
    plt.xlabel('Date')
    plt.ylabel('Oscillator Value')
    plt.legend(loc='upper left')

    # Plot ADX
    plt.subplot(4, 1, 3)
    plt.plot(data['ADX'], label='Average Directional Index (ADX)', color='brown', linewidth=1.2)
    plt.title('Average Directional Index (ADX)')
    plt.xlabel('Date')
    plt.ylabel('ADX Value')
    plt.legend(loc='upper left')

    # Plot CCI
    plt.subplot(4, 1, 4)
    plt.plot(data['CCI'], label='Commodity Channel Index (CCI)', color='magenta', linewidth=1.2)
    plt.title('Commodity Channel Index (CCI)')
    plt.xlabel('Date')
    plt.ylabel('CCI Value')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.8)

    
    # Set window position using wm_geometry for Tkinter
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+850+0")  # Position it on the right side of the screen

    plt.show()

# At the end of your script, use plt.show() to block until all windows are closed
plt.show()  # This keeps all windows open until manually closed

