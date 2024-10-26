from data_processing import load_data, prepare_data
from model_operations import build_model, train_model, test_model
from predictor import predict_next_day
from visualization import plot_candlestick_chart, plot_boxplot
import matplotlib.pyplot as plt

# Set key parameters for the dataset and model
COMPANY = 'CBA.AX'
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'
TEST_START, TEST_END = '2023-08-02', '2024-07-02'
PREDICTION_DAYS = 60
FEATURE_COLUMNS = ["Close", "Volume"]
NAN_METHOD, FILL_VALUE = 'ffill', 0
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-01-01'
RANDOM_SPLIT = False
USE_CACHE = True
CACHE_DIR = 'data_cache'

# Load, process, and split data
data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)

# Build and train the model
model = build_model((x_train.shape[1], len(FEATURE_COLUMNS)), num_layers=4, layer_type='BiLSTM', layer_size=100, dropout_rate=0.3)
train_model(model, x_train, y_train)
predicted_prices = model.predict(x_test)
predicted_prices = scalers["Close"].inverse_transform(predicted_prices)

# Plot actual vs predicted prices
y_test_unscaled = scalers["Close"].inverse_transform(y_test.reshape(-1, 1))
plt.plot(y_test_unscaled, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Display additional visualizations
plot_candlestick_chart(data, company=COMPANY, n_days=30)
plot_boxplot(data, company=COMPANY, n_days=90)

# Predict next day price based on the last sequence
last_sequence = x_test[-1].reshape(1, PREDICTION_DAYS, len(FEATURE_COLUMNS))
prediction = predict_next_day(model, last_sequence, scalers["Close"], PREDICTION_DAYS)
print(f"Prediction: {prediction}")
