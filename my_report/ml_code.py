import cleaning
df = cleaning.clean("C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv")

import statsmodels.tsa.arima.model.ARIMA
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())