import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def sta_plt(file_path, y, window=12):
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
