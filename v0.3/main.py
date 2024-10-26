from data_processing import load_data, prepare_data
from model_operations import build_model, train_model, test_model
from predictor import predict_next_day
from visualization import plot_candlestick_chart, plot_boxplot
import matplotlib.pyplot as plt

# COMPANY: Stock ticker symbol of the company to analyze
COMPANY = 'CBA.AX'

# TRAIN_START, TRAIN_END: Date range for training data
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'

# TEST_START, TEST_END: Date range for testing data
TEST_START, TEST_END = '2023-08-02', '2024-07-02'

# PREDICTION_DAYS: Number of days for input sequence (historical data)
PREDICTION_DAYS = 60

# FEATURE_COLUMNS: Dataset columns to scale and use in training
FEATURE_COLUMNS = ["Close", "Volume"]

# NAN_METHOD: Method to handle missing data (NaN values)
# FILL_VALUE: Value used to fill NaNs if 'fill' method is chosen
NAN_METHOD, FILL_VALUE = 'ffill', 0

# SPLIT_METHOD: Method for training/testing split ('ratio' or 'date')
# SPLIT_RATIO: Proportion of data for training if 'ratio' is selected
# SPLIT_DATE: Date to split data if 'date' method is selected
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-01-01'

# RANDOM_SPLIT: If True, data is split randomly; otherwise, sequentially
RANDOM_SPLIT = False

# USE_CACHE: If True, loads data from local cache if available
USE_CACHE = True

# CACHE_DIR: Directory to store/load cached data
CACHE_DIR = 'data_cache'

# Load and prepare data with options for caching, scaling, and splitting
data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)

# Process data: scale features, create input/output pairs, and split data
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)

# Build, train, and test the model
model = build_model((x_train.shape[1], len(FEATURE_COLUMNS)))  # Define model with input shape
train_model(model, x_train, y_train)  # Train the model with training data
predicted_prices = model.predict(x_test)  # Predict prices on the test set
predicted_prices = scalers["Close"].inverse_transform(predicted_prices)  # Rescale predictions to original values

# Rescale the true test prices for comparison
y_test_unscaled = scalers["Close"].inverse_transform(y_test.reshape(-1, 1))

# Plotting the actual vs. predicted prices as a line chart
plt.plot(y_test_unscaled, color="black", label=f"Actual {COMPANY} Price")  # Plot actual prices
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")  # Plot predicted prices
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Additional Visualization: Plot candlestick chart for the last 30 days
plot_candlestick_chart(data, company=COMPANY, n_days=30)

# Additional Visualization: Boxplot for multiple price segments over the last 90 days
plot_boxplot(data, company=COMPANY, n_days=90)

# Predict the next day's price based on the most recent test sequence
last_sequence = x_test[-1].reshape(1, PREDICTION_DAYS, len(FEATURE_COLUMNS))  # Reshape to match model input
prediction = predict_next_day(model, last_sequence, scalers["Close"], PREDICTION_DAYS)  # Predict next day price
print(f"Prediction: {prediction}")  # Display the predicted price
