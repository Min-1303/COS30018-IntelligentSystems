# File: experiments.py
# Purpose: Test LSTM, GRU, and BiLSTM models with various hyperparameters for stock prediction.
# Saves results (loss, time) to `experiment_results.csv`.

import time
import pandas as pd
from model_operations import build_model, train_model
from data_processing import load_data, prepare_data


# Model configurations
# These lists define the different model architectures (types of RNNs) and hyperparameters 
# (number of layers, units per layer, epochs, and batch size) that will be tested in the experiments.
model_types = ['LSTM', 'GRU', 'BiLSTM']  # Types of RNNs to experiment with
layers_config = [2, 3, 4]  # Different number of layers to experiment with
units_config = [50, 100, 150]  # Different numbers of units per layer to experiment with
epochs_config = [25, 50]  # Different numbers of training epochs to experiment with
batch_size_config = [32, 64]  # Different batch sizes for training

# Results list to store the outcome of each experiment
# Each configuration's results (e.g., losses, training time) will be appended to this list.
results = []

# Load and prepare data
# Define the stock symbol, date range, and features to be used for training and testing.
COMPANY = 'CBA.AX'  # Stock ticker symbol (Commonwealth Bank of Australia)
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'  # Training data date range
TEST_START, TEST_END = '2023-08-02', '2024-07-02'  # Testing data date range
FEATURE_COLUMNS = ["Close", "Volume"]  # The features to use for prediction (closing price and volume)
PREDICTION_DAYS = 60  # Number of days to predict into the future

# Handling missing data by forward-filling (ffill) and setting NaN values to 0.
NAN_METHOD, FILL_VALUE = 'ffill', 0  
# Split the dataset into training and testing sets based on a ratio (80% train, 20% test).
SPLIT_METHOD = 'ratio'  
SPLIT_RATIO = 0.8  # 80% training data, 20% test data
RANDOM_SPLIT = False  # Data is not shuffled randomly
USE_CACHE = True  # Use cached data if available to speed up loading times
CACHE_DIR = 'data_cache'  # Directory to store cached data

# Load the data using the `load_data` function from the `data_processing.py` module.
# The data is then prepared (split into training and test sets, and scaled) using the `prepare_data` function.
data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         random_split=RANDOM_SPLIT)

# Loop over all configurations of models and hyperparameters
# For each combination of model type, number of layers, units per layer, epochs, and batch size, 
# the model will be built, trained, and evaluated.
for model_type in model_types:
    for num_layers in layers_config:
        for layer_size in units_config:
            for epochs in epochs_config:
                for batch_size in batch_size_config:
                    # Print the current configuration being tested to keep track of the progress.
                    print(f"Training {model_type} with layers = {num_layers}, units = {layer_size}, epochs = {epochs}, batch size = {batch_size}")

                    # Build the model using the `build_model` function from `model_operations.py`.
                    # The input shape is based on the training data's shape.
                    input_shape = (x_train.shape[1], x_train.shape[2])
                    model = build_model(input_shape=input_shape, num_layers=num_layers, layer_type=model_type, layer_size=layer_size)

                    # Measure the time it takes to train the model.
                    start_time = time.time()
                    trained_model = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
                    training_time = time.time() - start_time  # Calculate the training duration.

                    # Evaluate the model's performance on both the training and validation (test) data.
                    train_loss = trained_model.evaluate(x_train, y_train, verbose=0)  # MSE on training data
                    val_loss = trained_model.evaluate(x_test, y_test, verbose=0)  # MSE on validation (test) data

                    # Save the experiment's results (model type, hyperparameters, loss, and training time) to the results list.
                    results.append({
                        'Model Type': model_type,  # Type of the model (LSTM, GRU, or BiLSTM)
                        'Layers': num_layers,  # Number of layers in the model
                        'Units per 1 layer': layer_size,  # Number of units in each layer
                        'Epochs': epochs,  # Number of epochs used for training
                        'Batch Size': batch_size,  # Batch size used for training
                        'Training Loss': train_loss,  # MSE on training data
                        'Validation Loss': val_loss,  # MSE on test data
                        'Time Taken (s)': training_time  # Time taken for training (in seconds)
                    })

# Convert the results list to a DataFrame for easy analysis.
data = pd.DataFrame(results)

# Save the results to a CSV file for future reference and analysis.
data.to_csv("experiment_results.csv", index=False)

# Print the final results to the console for quick review.
print(data)
