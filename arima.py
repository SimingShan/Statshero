import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from cleaning import clean

def arima_model(user_input, y):
    # Perform auto ARIMA to determine optimal p and q values
    series = clean(user_input)[y]
    model = auto_arima(series, suppress_warnings=True, error_action='ignore')

    # Fit ARIMA model with optimal p and q values
    model.fit(series)

    # Get the predicted values and residuals
    pred = model.predict_in_sample()
    residuals = series - pred
    summary = model.summary()

    # Plot the predicted values and residuals
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Actual')
    plt.plot(pred, label='Predicted')
    plt.plot(residuals, label='Residuals')
    plt.legend()
    plt.title('ARIMA Model')
    plt.show()
    return model, pred, residuals, summary



