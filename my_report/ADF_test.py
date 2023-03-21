import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def is_stationary(file_path, y, window=12):
    data = pd.read_csv(file_path)
    series = data[y]

    # Calculate rolling mean and rolling standard deviation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Perform Augmented Dickey-Fuller test
    result = adfuller(series)
    p_value = result[1]
    if p_value < 0.05:
        message_st = "The time series is stationary"
    else:
        message_st = "The time series is not stationary due to its"
        if rolling_mean.isnull().sum() < len(rolling_mean):
            if not np.all(np.isclose(rolling_mean.dropna().values, rolling_mean.dropna().values[0])):
                message_st = message_st + " non-stationary mean"
        if rolling_std.isnull().sum() < len(rolling_std):
            if not np.all(np.isclose(rolling_std.dropna().values, rolling_std.dropna().values[0])):
                message_st = message_st + " non-stationary standard deviation"

    return message_st
