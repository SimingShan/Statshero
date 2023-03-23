import cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = cleaning.clean('C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv')

x = df[['date']]
y = df['humidity']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.scatter(x_test, y_test, color='salmon')
plt.plot(x_test, y_pred, color='tomato', linewidth=2)
plt.title('Linear Regression Model')

residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(x_test, residuals, color='black')
plt.title('Residual Plot')
plt.show()