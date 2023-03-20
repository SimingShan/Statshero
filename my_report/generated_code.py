import cleaning
import matplotlib.pyplot as plt

df = cleaning.clean("C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv")

plt.plot(df["Date"], df["High"])
plt.title("Nasdaq High Prices Over Time")
plt.xlabel("Date")
plt.ylabel("High Price")

plt.show()