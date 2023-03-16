# Import all necessary packages
import openai
import runpy
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
path4 = "C:/Users/int_shansiming/Desktop/Prediction/salary.xlsx"

# Set up the parameters for the GPT-3 API
model = "text-davinci-002"
temperature_1 = 0.5
temperature_2 = 1
max_tokens = 3000

# Ask for file location
user_input_file = input("Enter the file location:")

# import the data
try:
    user_data = pd.read_csv(user_input_file)
except ValueError:
    user_data = pd.read_excel(user_input_file)

# Then get the column names
col_name = user_data.columns.tolist()

# Ask user for input
user_input_1 = input("Enter your request")

# Language setting
user_language = input("In what language do you wish your report to be?")

# Ask for features if the user ask for a plot
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    features_x = input(f"Select your x from ({col_name}): ")
    features_y = input(f"Select your y from ({col_name}): ")
    method = input("Enter your desired plot type")
else:
    user_input_1 = user_input_1

# Introduce the plot type
# Check if the input contains any keywords
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    prompt = f'''Introduce {method} and explain how they are used in data analysis  in {user_language}.'''

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_1,
    )

print(f"{method}, {response.choices[0].text.strip()}")

# Check if the input contains any keywords
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    prompt = f'''
   Generate Python code to accomplish the following tasks:
1. Import cleaning.py and use the cleaning.clean() function to import and clean the dataset from {user_input_file}, save as 'df'.
2. Import matplotlib.pyplot as plt and create a {method} to display the relationship between x = {features_x} and y = {features_y}.
3. Add a title to the graph using the Matplotlib library.
4. If there are two features in {features_y}, add a legend.
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

    runpy.run_path('generated_code.py')

else:
    prompt = f'''{user_input_1}, The file is from: {user_input_file}'''
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_1,
    )
    print(response.choices[0].text.strip())

# Using openai api to generate a comprehensive report
user_input_file = pd.read_csv(user_input_file)
prompt_text = f'''
Please provide a comprehensive data analysis report for the following dataset:

{user_input_file}

The report should include:

1. Overview of the dataset: Briefly describe the dataset (how many observations there are), its variables, and their meanings.
2. Descriptive statistics (mean, median, mode, standard deviation, etc.) for each variable
3. Identification of any potential outliers or missing values
4. Correlations between variables and their significance
5. Identify trends, patterns, correlations, or anomalies in the data and describe their significance.

And the report should be in {user_language}

Please present the findings in a clear, precise, and professional manner.
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

prompt_text = f'''
Please provide the required data analysis report in {user_language} for the following dataset:

{user_input_file}

The report should include:

1. Analyze {features_x} and {features_y} using relevant statistical learning techniques(like regression or ARIMA)
2. Build the model and display the result parameter, and explain its meaning(Do not show the code)
3. Recommendations for further analysis


Please present the findings in a clear, precise, and professional manner.
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
path4 = "C:/Users/int_shansiming/Desktop/Prediction/salary.xlsx"
