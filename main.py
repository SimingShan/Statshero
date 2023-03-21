# Import all necessary packages
import openai
import runpy
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
from statsmodels.tsa.stattools import adfuller
import cleaning

openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
path4 = "C:/Users/int_shansiming/Desktop/Prediction/elec.csv"

# ------------------------------------------
# Set up the parameters for the GPT-3 API
model = "text-davinci-002"
temperature_1 = 0.1
temperature_2 = 1
max_tokens = 3200

# ------------------------------------------
# Ask for file location
user_input_file = input("Enter the file location:");

# import the data
try:
    user_data = cleaning.clean(user_input_file)
except ValueError:
    user_data = cleaning.clean(user_input_file)

# Then get the column names
col_name = user_data.columns.tolist()

# Ask for features if the user ask for a plot
features_x = input(f"Select your x from {col_name}: ");
features_y = input(f"Select your y from {col_name}: ");
x_type = user_data[features_x].dtype
y_type = user_data[features_y].dtype
prompt = f'''Based on the name and the type of x and y, distinguish what is the best model for prediction
the name of x is {features_x}
the name of y is {features_y}
the type of x is {x_type}
the type of y is {y_type}
if x is date, then consider time series first
print only the name of the best model, no explainations needed
'''
response_text = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=1,
)
rec_ml = "The recommended model for these features is" + f'''{response_text.choices[0].text.strip()}'''
print(rec_ml)

# ------------------------------------------



print(" ")
print("\033[1m\033[4m\033[36mPlot\033[0m")
print(" ")

prompt = f'''
   Generate Python code to accomplish the following tasks:
1. Import cleaning.py and use the cleaning.clean({user_input_file}), save as 'df'.
2. Import matplotlib.pyplot as plt and create a plot to display the relationship between x = {features_x} and y = {features_y}.
3. make the size of the plot: plt.figure(figsize=(12, 6))
4. Add a title to the graph using the Matplotlib library.
5. Label the axes using appropriate units based on the names of the features.


Please provide the code without any additional comments or notes.
    '''
    # Generate code using the GPT-3 API
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature_1,
)

# Save the generated code to a file
with open("generated_code.py", "w") as f:
    f.write(response.choices[0].text.strip())

# Import the generated code as a module
import generated_code
runpy.run_path("generated_code.py")


# ------------------------------------------
# Using openai api to generate a comprehensive report
print(" ")
print("\033[1m\033[4m\033[36mIntroduction to the variables\033[0m")
print(" ")
prompt_text = f'''
Using the Excel file {user_input_file}, please provide a brief introduction to the following variables: 
{col_name}. 
Please only describe these variables and do not create any new ones. 
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

# --------------------------
# Generate the descriptive statistics
print(" ")
print("\033[1m\033[4m\033[36mDescriptive statistics\033[0m")
print(" ")
from Descriptive_statistics import des_chart

des_stats = des_chart(user_input_file)
print(tabulate(des_stats, headers='keys', tablefmt='github', showindex=True))

# Write a prompt that can read the descriptive statistics
prompt_des_stats = f'''
Here is a chart of descriptive statistics from the Excel file {user_input_file}:
{des_stats}
Please provide a detailed description and insights of the main characteristics and patterns in this summary chart
In a professional statistician's tongue.
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_des_stats,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

# --------------------------
# Generate the correlation relationship chart
print(" ")
print("\033[1m\033[4m\033[36mCorrelation\033[0m")
print(" ")
from Descriptive_statistics import cor_chart

cor_stats = cor_chart(user_input_file)
print(tabulate(cor_stats, headers='keys', tablefmt='github', showindex=True))

prompt_cor_stats = f'''
Here is a chart of correlation from the Excel file {user_input_file}:
{cor_stats}
Please provide a detailed description and insights of 
the main characteristics and patterns in this correlation chart,
then analyze its insights
In a professional statistician's tongue.
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_cor_stats,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

###

# --------------------------
# --------------------------
# If the data is time series, then whether stationary?
print(" ")
print("\033[1m\033[4m\033[36mTime Series Analysis\033[0m")
print(" ")
from ADF_test import is_stationary
from stationary_plot import sta_plt

# Generate rolling mean and sd plot
sta_plt(user_input_file, features_y)

# Generate if stationary True/False
message_st = is_stationary(user_input_file, features_y)

prompt_cor_stats = f'''
Here is a message about whether the Excel file {user_input_file} is stationary about feature {features_y}:
{message_st}
Please provide a detailed description about stationary,and analyze the meaning of the above message in context
in a neat and professional statistician's tongue.
The output should be in fowllowing format:
A stationary time series data means that:
The above graph indicates that the time series is:
'''
response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_cor_stats,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

# Perform ARIMA
from arima import arima_model
arima_model,arima_pred,arima_residuals,arima_summary = arima_model(user_input_file, features_y)
# Use AI to explain the model
prompt_arima = f'''Given an {arima_model} model
please explain the model in detail. Assume that the model has already been fit to a time series dataset.
Please explain the meaning and significance of each of these parameters 
Additionally, please explain how the model was fit to the data, 
and how the predict() and predict_in_sample() methods can be used to forecast future values of the time series.
Finally, please provide any additional insights or observations about the model that you feel would be helpful in understanding its behavior and performance.
'''
response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_arima,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

prompt_arima_2 = f'''Given a list of residuals from {arima_residuals}, explain how well an ARIMA model has performed:

The ARIMA model was used to forecast a time series. The model generated a list of residuals by subtracting the predicted values from the actual values. Please analyze the list of residuals and provide a detailed report on how well the ARIMA model has performed. Specifically, please answer the following questions:

1. What is the mean of the residuals?
2. What is the standard deviation of the residuals?
3. Are the residuals normally distributed? 
4. Is there any evidence of autocorrelation in the residuals? 
5. Is there any evidence of heteroscedasticity in the residuals? 
'''
response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_arima_2,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

print(arima_summary)
prompt_arima_2 = f'''Please explain the results table for a SARIMAX model generated using the auto_arima function in Python. 
The table is displayed in the following format:
{arima_summary}
Please provide a detailed explanation of the table, including what each column and row represents, 
and the significance of the coefficients and test statistics. 
Based on the value of each coefficients and test statistics, explain how well the model is performing
Additionally, please explain any other relevant information,
such as the model order and how it was determined, and any assumptions that were made in fitting the model.
'''
response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_arima_2,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
