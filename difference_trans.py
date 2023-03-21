import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cleaning
from ADF_test import is_stationary
from cleaning import clean
from statsmodels.tsa.stattools import adfuller

def stationary(series, window=12):
    # Calculate rolling mean and rolling standard deviation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    # Perform Augmented Dickey-Fuller test
    result = adfuller(series)
    p_value = result[1]
    if p_value < 0.05:
        return True
    else:
        return False


def difference_equ(user_input,y):
    data = cleaning.clean(user_input)
    series = data[y]
    stationary_or_not = stationary(series)
    n = 0
    while not stationary_or_not:
        series = series.diff(periods=1)
        series = series.dropna()
        n = n + 1
        stationary_or_not = stationary(series)
    return series, n

def diff_plt(user_input, y):
    data = cleaning.clean(user_input)
    series, n = difference_equ(user_input,y)
    plt.figure(figsize=(12, 6))
    plt.plot(data[y])
    plt.plot(series)
    plt.legend(['Original', 'Differenced'])
    plt.show()
    print(f'''The series was difference transformed by {n} times(n = {n}), the above graph \
shows that the series is now stationary''')
    return series, n










