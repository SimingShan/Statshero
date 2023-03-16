import cleaning
df = cleaning.clean('C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv')
import matplotlib.pyplot as plt
plt.scatter(df['humidity'], df['meanpressure'])
plt.title('Relationship Between Humidity and Mean Pressure')
plt.xlabel('Humidity (%)')
plt.ylabel('Mean Pressure (hPa)')
plt.show()