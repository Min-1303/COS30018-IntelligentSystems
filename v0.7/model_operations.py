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


# Plot Predictions
def plot_predictions(y_test_unscaled, xgb_predictions, svr_predictions, rf_predictions):
    """
    Plot the actual prices vs. predicted prices from XGBoost, SVR, and Random Forest models.
    """
    plt.figure(figsize=(12, 7))

    # Actual prices
    plt.plot(y_test_unscaled, color="black", linewidth=2, label="Actual Prices")

    # XGBoost predictions
    plt.plot(xgb_predictions, color="blue", linestyle='--', label="XGBoost Predictions")

    # SVR predictions
    plt.plot(svr_predictions, color="green", linestyle='--', label="SVR Predictions")

    # Random Forest predictions
    plt.plot(rf_predictions, color="red", linestyle='-', label="Random Forest Predictions")

    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


# Plot RSI and MACD
def plot_rsi_macd(data):
    rsi_values = data['RSI'].values
    macd_values = data['MACD'].values * 10  # Scaling MACD by a factor of 10
    theta = np.linspace(0, 2 * np.pi, len(rsi_values))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, rsi_values, linestyle='--', color='green', label="RSI")
    ax.plot(theta, macd_values, linestyle='-', color='orange', label="MACD")

    plt.title("RSI & MACD Indicators (Polar Chart)")
    plt.legend(loc="upper right")
    plt.show()
