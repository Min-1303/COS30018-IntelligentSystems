import pandas as pd
import warnings
from data_processing import load_data, prepare_data, load_macro_data
from model_operations import train_and_predict_models, plot_predictions, plot_rsi_macd  # Include the plot_rsi_macd function

# Configuration Parameters
COMPANY = 'CBA.AX'
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'
PREDICTION_DAYS = 60
FEATURE_COLUMNS = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# === Data Loading and Preparation ===
def load_and_prepare_data():
   
    stock_data = load_data(COMPANY, TRAIN_START, TRAIN_END)
    macro_data = load_macro_data()
    macro_data = macro_data.resample('D').ffill()

    data = stock_data.join(macro_data, how='left').ffill()

    print(f"Data columns after joining macroeconomic data: {data.columns}")

    # Prepare data for training
    x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                             split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO)
    return x_train, y_train, x_test, y_test, scalers, data


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scalers, data = load_and_prepare_data()

    # Train models and get predictions
    xgb_predictions, svr_predictions, rf_predictions, y_test_unscaled = train_and_predict_models(
        x_train, y_train, x_test, y_test, scalers
    )

    # Plot the results (XGBoost, SVR, Random Forest)
    plot_predictions(y_test_unscaled, xgb_predictions, svr_predictions, rf_predictions)

    # Plot RSI and MACD
    plot_rsi_macd(data)
