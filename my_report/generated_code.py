import cleaning
import matplotlib.pyplot as plt

df = cleaning.clean('C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv')

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['wind_speed'])
plt.title('Wind Speed over Time')
plt.xlabel('Date')
plt.ylabel('Wind Speed (m/s)')
plt.show()