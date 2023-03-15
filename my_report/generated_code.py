import pandas as pd

df = pd.read_csv('C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv')

df.head()

df.describe()

import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plt.plot( 'date', 'meantemp', data=df, marker='', color='blue', linewidth=2)
plt.plot( 'date', 'humidity', data=df, marker='', color='olive', linewidth=2)
plt.plot( 'date', 'wind_speed', data=df, marker='', color='red', linewidth=2)
plt.plot( 'date', 'meanpressure', data=df, marker='', color='purple', linewidth=2)

plt.legend()

plt.title("Daily Delhi Climate Test")

plt.show()