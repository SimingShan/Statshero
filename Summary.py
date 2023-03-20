import openai
import runpy
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
path4 = "C:/Users/int_shansiming/Desktop/Prediction/salary.xlsx"


# Set up the parameters for the GPT-3 API
model = "text-davinci-002"
temperature_1 = 0.1
temperature_2 = 1
max_tokens = 3200

# Ask for file location
user_input_file = input("Enter the file location:");

# import the data
try:
    user_data=pd.read_csv(user_input_file)
except ValueError:
    user_data=pd.read_excel(user_input_file)

# Then get the column names
col_name = user_data.columns.tolist()

######################################################################

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

# Generate the descriptive statistics
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


# Generate the correlation relationship chart
from Descriptive_statistics import cor_chart
cor_stats = cor_chart(user_input_file)
print(tabulate(cor_stats, headers='keys', tablefmt='github', showindex=True))

prompt_cor_stats = f'''
Here is a chart of correlation from the Excel file {user_input_file}:
{cor_stats}
Please provide a detailed description and insights of 
the main characteristics and patterns in this correlation chart
In a professional statistician's tongue.
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_cor_stats,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')
