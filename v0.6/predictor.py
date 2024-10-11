# File: predictor.py
# Purpose: This module is responsible for making predictions using the trained
# model. It processes the most recent data and outputs the predicted stock price
# for the next day.

import numpy as np

def predict_next_day(model, last_sequence, scaler, prediction_days):
    """
    Predict the stock price for the next day based on the last sequence of data.
    
    Args:
        model: Trained machine learning model to make predictions.
        last_sequence (numpy.ndarray): Sequence of the most recent stock data, used as input for the model.
        scaler: Scaler object used to reverse the scaling of the predicted stock price.
        prediction_days (int): Number of past days considered in the input sequence to predict the next day's price.
    
    Returns:
        The predicted stock price for the next day, in its original scale.
    """
    # Use only the last 'prediction_days' of stock data from the input sequence
    real_data = last_sequence[-prediction_days:]

    # Model predicts the stock price for the next day
    prediction = model.predict(real_data)

    # Reverse the scaling applied earlier to get the actual predicted price
    return scaler.inverse_transform(prediction)


def multistep(model, last_sequence, scaler, steps):
    """
    Predict stock prices for multiple days into the future (multistep prediction).
    
    Args:
        model: Trained model used for prediction.
        last_sequence (numpy.ndarray): Last sequence of stock data, serving as input for the first prediction.
        scaler: Scaler object to reverse the scaling of the predicted data.
        steps (int): Number of future days to predict.
    
    Returns:
        List of predicted stock prices for the next 'steps' days.
    """
    predictions = []  # Store predictions for each day
    current_sequence = last_sequence  # Initialize the input sequence with the latest available data

    # Loop over the number of days to predict
    for _ in range(steps):
        # Model predicts the next day's stock price
        prediction = model.predict(current_sequence)

        # Reverse the scaling to get the actual predicted price
        prediction = scaler.inverse_transform(prediction)

        # Add the prediction for the current day to the list
        predictions.append(prediction[0, 0])

        # Prepare for the next prediction by updating the sequence with the latest prediction
        # Create a placeholder for the next input sequence, shifting the existing sequence by 1 day
        new_data_point = np.zeros((1, 1, current_sequence.shape[2]))  # A new day of data
        new_data_point[0, 0, 0] = prediction[0, 0]  # Insert the predicted value into the new sequence

        # Shift the current sequence and add the new data point at the end
        current_sequence = np.append(current_sequence[:, 1:, :], new_data_point, axis=1)

    return predictions  # Return the list of predictions


def multivariate(model, data, feature_columns, scaler, prediction_days):
    """
    Predict stock price for a future day using multiple features (multivariate prediction).
    
    Args:
        model: Trained model used for prediction.
        data (pandas.DataFrame): Historical stock data with multiple features (open, close, volume, etc.).
        feature_columns (list): List of columns (features) to use for prediction (e.g., open, close prices).
        scaler: Scaler object used to reverse the scaling of the predicted data.
        prediction_days (int): Number of days of historical data to use for the prediction.
    
    Returns:
        The predicted closing price for the specified future day.
    """
    # Extract the last 'prediction_days' of data for the specified feature columns
    last_sequence = data[-prediction_days:][feature_columns].values

    # Scale the data to normalize the values before making predictions
    last_sequence = scaler.transform(last_sequence)

    # Reshape the sequence to the format expected by the model: [batch_size, time_steps, num_features]
    last_sequence = last_sequence.reshape(1, prediction_days, len(feature_columns))

    # Use the model to predict the stock price based on the input sequence
    prediction = model.predict(last_sequence)

    # Create a placeholder to store the predicted value
    reshaped_prediction = np.zeros((1, len(feature_columns)))
    reshaped_prediction[0, 0] = prediction[0, 0]  # Assign the predicted closing price to the first feature

    # Reverse the scaling of the prediction to get the actual predicted price
    return scaler.inverse_transform(reshaped_prediction)[0, 0]


def multivariate_multistep(model, data, feature_columns, scaler, prediction_days, steps):
    """
    Predict stock prices for multiple days into the future using multiple features (multivariate and multistep prediction).
    Args:
        model: Trained model used for prediction.
        data (pandas.DataFrame): Historical stock data with multiple features.
        feature_columns (list): List of columns (features) used for prediction (e.g., open, close prices).
        scaler: Scaler object used to reverse the scaling of the predicted data.
        prediction_days (int): Number of days of historical data to use as input for predictions.
        steps (int): Number of days to predict into the future.
    Returns:
        A list of predicted closing prices for the next 'steps' days.
    """
    predictions = []  # Store predictions for each future day

    # Extract and scale the last 'prediction_days' worth of data for the specified feature columns
    last_sequence = data[-prediction_days:][feature_columns].values
    last_sequence = scaler.transform(last_sequence)

    # Reshape the sequence to the required format [batch_size, time_steps, num_features]
    current_sequence = last_sequence.reshape(1, prediction_days, len(feature_columns))

    # Loop over the number of steps (days) to predict
    for _ in range(steps):
        # Model predicts the closing price for the next day
        prediction = model.predict(current_sequence)

        # Create a placeholder for the predicted values, with the same shape as the feature columns
        reshaped_prediction = np.zeros((1, len(feature_columns)))

        # Assign the predicted closing price to the correct feature column
        reshaped_prediction[0, 3] = prediction[0, 0]  # Assuming the closing price is in the 4th column

        # Reverse the scaling to get the actual predicted closing price
        prediction_rescaled = scaler.inverse_transform(reshaped_prediction)

        # Add the predicted closing price to the list of predictions
        predictions.append(prediction_rescaled[0, 3])

        # Update the current sequence with the new predicted closing price for the next prediction
        new_data_point = np.zeros((1, 1, current_sequence.shape[2]))  # A new day of data
        new_data_point[0, 0, 3] = prediction[0, 0]  # Update the closing price for the new sequence
        current_sequence = np.append(current_sequence[:, 1:, :], new_data_point, axis=1)

    return predictions  # Return the list of predicted closing prices for multiple days
