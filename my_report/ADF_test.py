import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def is_stationary(file_path, y, window=12):
    data = pd.read_csv(file_path)
    series = data[y]

    # Calculate rolling mean and rolling standard deviation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Plot the original series, rolling mean, and rolling standard deviation
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original Series')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Original Series with Rolling Mean and Standard Deviation')
    plt.show()

    # Perform Augmented Dickey-Fuller test
    result = adfuller(series)
    p_value = result[1]
    if p_value < 0.05:
        message = "Data is stationary"
    else:
        message = "Data is not stationary,"
        if rolling_mean.isnull().sum() < len(rolling_mean):
            if not np.all(np.isclose(rolling_mean.dropna().values, rolling_mean.dropna().values[0])):
                message = message +  "due to non-constant mean"
        if rolling_std.isnull().sum() < len(rolling_std):
            if not np.all(np.isclose(rolling_std.dropna().values, rolling_std.dropna().values[0])):
                message = message + "and due to non-constant variance"
    return message