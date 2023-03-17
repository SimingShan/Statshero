import cleaning
df = cleaning.clean('C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv')

import matplotlib.pyplot as plt
plt.scatter(df['humidity'], df['wind_speed'])
plt.title('Relationship between Humidity and Wind Speed')
plt.xlabel('Humidity (%)')
plt.ylabel('Wind Speed (km/h)')