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
from cleaning import clean
import importlib

openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"

# ------------------------------------------
# Set up the parameters for the GPT-3 API
model_text = "text-davinci-002"
model_code = "text-davinci-002"
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

from request_matching import parse_user_input, select_variables_and_model

while True:
    # Enter request
    user_request = input("Please enter your request: ")

    # Get the desired variables from the input request
    parsed_msg = parse_user_input(user_input_file,user_request)
    # selected_v = select_variables_and_model(user_input_file, user_request)
    selected_v = []
    for v in col_name:
        if v in parsed_msg:
            selected_v.append(v)

    # Check if the selected_v set has more than one element
    if len(selected_v) > 1:
        break
    else:
        print("Please provide a request with at least two variables.")
print(col_name)
selected_type = []
for v in selected_v:
    v_type = user_data[v].dtype
    selected_type.append(v_type)

features_y = input(f"Select your y from {selected_v}: ");
y_index = selected_v.index(features_y)
selected_v.pop(y_index)
y_type = selected_type[y_index]
selected_type.pop(y_index)

success = False
attempts = 0
while not succuss and attempts <1:
    try:
        prompt = f'''Based on the name and the type of x and y, and the request {parsed_msg}
        ,distinguish what is the best statistical model for user's request
        the name of x is {selected_v}
        the name of y is {features_y}
        the type of x is {selected_type} accordingly
        the type of y is {y_type}
        choose one suitable model based on bias-variance trade-off
        print only the name of one suitable modelchoose the model based on bias-variance trade-off, no explainations needed
        '''
        response_text = openai.Completion.create(
            engine=model_text,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        rec_ml = "The recommended model for these features is " + f'''{response_text.choices[0].text.strip()}'''
        prompt = f'''Write Python code to:
        Import the necessary packages, including the custom cleaning.py module.
        Load the dataset using the cleaning.clean function, providing the user's input file {user_input_file} as an argument, and save it as 'df'.
        from sklearn.model_selection import train_test_split
        Split the dataset into training and testing sets using the dependent variable y represented by {features_y} and the independent variables x represented by {selected_v}.
        Fit a model using the {rec_ml} method on the training set.
        Generate predictions using the fitted model on the testing set.
        Create am appropriate plot of the original data and the predictions using the 'matplotlib.pyplot' library, set the figsize by plt.figure(figsize=(12, 6)).
        Add a title to the graph using the Matplotlib library.
        Create an residual plot showing the difference between the predictions and testing y values.
        Exclude any comments or notes from the code, generate code only.
        The color shoule be chosen from ('salmon','tomato','black')
        Follow the above instructions step by step, and do not miss any instructions.
        '''

        # Generate code using the GPT-3 API
        response = openai.Completion.create(
            engine=model_code,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature_1,
        )

        # Save the generated code to a file
        with open("gen_ml_code.py", "w") as f:
            f.write(response.choices[0].text.strip())

        path_to_script = "gen_ml_code.py"
        graph_gen = runpy.run_path(path_to_script)
        success = True
    except Exception as e:
        print(f"Error: {e}")
        attempts += 1


# analyze the relationship between humidity mean teamperature and wind speed, so that i can predict humidity
# C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv