import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def candlestick_chart(data, company, n_days=1):
    """
    Plot a candlestick chart for the given stock market data.
    Each candlestick represents 'n_days' of trading data.
    
    Arguments:
    - data: DataFrame containing stock market data with columns like 'Open', 'High', 'Low', 'Close', and 'Volume'.
    - company: A string representing the stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    - n_days: Integer specifying the number of trading days each candlestick should represent (default is 1).
    """
    
    # If n_days is greater than 1, resample the data to group it into larger time intervals
    if n_days > 1:
        # Resampling the data to aggregate based on 'n_days'
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first',  # The first opening price in the period
            'High': 'max',    # The highest price in the period
            'Low': 'min',     # The lowest price in the period
            'Close': 'last',  # The last closing price in the period
            'Volume': 'sum'   # The total volume of stocks traded in the period
        }).dropna()  # Remove any rows with NaN values that might have been created during resampling
    else:
        # If n_days is 1, use the original data without resampling
        data_resampled = data
    
    # Plotting the candlestick chart using the resampled data
    mpf.plot(data_resampled, type='candle', title=f'{company} Candlestick Chart', style='charles', volume=True)


def boxplot_chart(data, company, column='Close', n_days=1):
    """
    Plot multiple boxplot charts for the given stock market data.
    Each boxplot shows the distribution of data over a moving window of 'n_days'.
    
    Arguments:
    - data: DataFrame containing stock market data.
    - company: A string representing the stock ticker symbol.
    - column: The column of the data to be visualized (default is 'Close' price).
    - n_days: Integer specifying the number of consecutive trading days in the moving window (default is 1).
    """
    
    # Calculate the rolling mean over a window of 'n_days' and drop any resulting NaN values
    rolling_data = data[column].rolling(window=n_days).mean().dropna()
    
    # Create a list of rolling data segments to use for each boxplot
    boxplot_data = [rolling_data[i:i + n_days] for i in range(0, len(rolling_data), n_days)]
    
    # Create a figure for the boxplot with a specific size
    plt.figure(figsize=(12, 6))
    
    # Set the title of the boxplot chart
    plt.title(f'{company} Boxplot Chart')
    
    # Generate the boxplot with various customization options
    plt.boxplot(boxplot_data, patch_artist=True, showmeans=True)
    
    # Label the x-axis with the rolling periods and y-axis with 'Closing Price'
    plt.xlabel(f'Rolling {n_days}-Day Period')
    plt.ylabel('Closing Price')
    
    # Customize the x-axis labels to reflect the date ranges for each boxplot
    plt.xticks(ticks=range(1, len(boxplot_data) + 1), labels=[f'{i*n_days+1}-{(i+1)*n_days}' for i in range(len(boxplot_data))])
    
    # Add a grid to the chart for better readability
    plt.grid(True)


def plot_multistep_prediction(actual, predicted, company, steps):
    """
    Plot multistep stock price predictions vs. actual prices.
    
    Args:
        actual (array-like): The actual stock prices.
        predicted (array-like): The predicted stock prices.
        company (str): Company name or ticker symbol.
        steps (int): The number of steps (days) predicted.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:steps], color="black", label=f"Actual {company} Price")
    plt.plot(range(len(predicted)), predicted, color="green", label=f"Multistep Predicted {company} Price", linestyle='--')
    plt.title(f"{company} Multistep Share Price Prediction", fontsize=16)
    plt.xlabel("Steps into Future", fontsize=12)
    plt.ylabel(f"{company} Share Price", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multivariate_prediction(actual, predicted, company):
    """
    Plot multivariate stock price predictions vs. actual prices.
    
    Args:
        actual (array-like): The actual stock prices.
        predicted (array-like): The predicted stock prices.
        company (str): Company name or ticker symbol.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:5], color="black", label=f"Actual {company} Price")
    plt.plot(range(5), predicted, color="green", label=f"Multivariate Predicted {company} Price", linestyle='--')
    plt.title(f"{company} Multivariate Share Price Prediction", fontsize=16)
    plt.xlabel("Steps into Future", fontsize=12)
    plt.ylabel(f"{company} Share Price", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multivariate_multistep_prediction(actual, predicted, company, steps):
    """
    Plot multivariate multistep stock price predictions vs. actual prices.
    
    Args:
        actual (array-like): The actual stock prices.
        predicted (array-like): The predicted stock prices.
        company (str): Company name or ticker symbol.
        steps (int): The number of steps (days) predicted.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:steps], color="black", label=f"Actual {company} Price")
    plt.plot(range(len(predicted)), predicted, color="green", label=f"Multivariate Multistep Predicted {company} Price", linestyle='--')
    plt.title(f"{company} Multivariate Multistep Share Price Prediction", fontsize=16)
    plt.xlabel("Steps into Future", fontsize=12)
    plt.ylabel(f"{company} Share Price", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
