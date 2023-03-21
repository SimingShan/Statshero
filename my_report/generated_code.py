import cleaning
import matplotlib.pyplot as plt

df = cleaning.clean('C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv')

plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['meantemp'])
plt.title('Mean Temperature in Delhi')
plt.xlabel('Date')
plt.ylabel('Mean Temperature (Celsius)')
plt.show()