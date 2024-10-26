from data_processing import load_data, prepare_data
from model_operations import build_model, train_model, test_model
from predictor import predict_next_day
import matplotlib.pyplot as plt

# Parameters defining the company and data range for stock analysis
COMPANY = 'CBA.AX'
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'  # Training period
TEST_START, TEST_END = '2023-08-02', '2024-07-02'  # Testing period
PREDICTION_DAYS = 60  # Days of historical data for each input sequence

# Features to use for model input and methods for data handling
FEATURE_COLUMNS = ["Close", "Volume"]
NAN_METHOD, FILL_VALUE = 'ffill', 0

# Splitting method parameters for training/testing sets
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-01-01'
RANDOM_SPLIT = False

# Data caching options
USE_CACHE = True
CACHE_DIR = 'data_cache'

# Load, process, and split data
data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)

# Build and train the model
model = build_model((x_train.shape[1], len(FEATURE_COLUMNS)))
train_model(model, x_train, y_train)

# Predict on the testing data and inverse scale predictions
predicted_prices = model.predict(x_test)
predicted_prices = scalers["Close"].inverse_transform(predicted_prices)

# Plot the results: actual vs predicted prices
y_test_unscaled = scalers["Close"].inverse_transform(y_test.reshape(-1, 1))
plt.plot(y_test_unscaled, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Predict the next day's price based on the last test sequence
last_sequence = x_test[-1].reshape(1, PREDICTION_DAYS, len(FEATURE_COLUMNS))
prediction = predict_next_day(model, last_sequence, scalers["Close"], PREDICTION_DAYS)
print(f"Prediction: {prediction}")
