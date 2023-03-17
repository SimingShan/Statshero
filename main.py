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
temperature_1 = 0.1
temperature_2 = 1
max_tokens = 3200

# Ask for file location
user_input_file = input("Enter the file location:");

# import the data
try:
    user_data = pd.read_csv(user_input_file)
except ValueError:
    user_data = pd.read_excel(user_input_file)

# Then get the column names
col_name = user_data.columns.tolist()

# Ask user for input
user_input_1 = input("Enter your request");

# Language setting
user_language = input("In what language do you wish your report to be?");

# Ask for features if the user ask for a plot
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    features_x = input(f"Select your x from ({col_name}): ");
    features_y = input(f"Select your y from ({col_name}): ");
    method = input("Enter your desired plot type");
else:
    user_input_1 = user_input_1

# Building a model?
ml_model = input("What method do you wish to build your model?")

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
# Unsolved problem :4. If there are two features in {features_y}, add a legend.
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    prompt = f'''
   Generate Python code to accomplish the following tasks:
1. Import cleaning.py and use the cleaning.clean({user_input_file}), save as 'df'.
2. Import matplotlib.pyplot as plt and create a {method} to display the relationship between x = {features_x} and y = {features_y}.
3. Add a title to the graph using the Matplotlib library.
4. Label the axes using appropriate units based on the names of the features.

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

prompt_text = f'''
Please provide a comprehensive data analysis report for the following dataset:

{user_input_file}
The report should include:
Overview of the dataset: Briefly describe the dataset (how many observations there are), its variables, and their meanings.
Descriptive statistics: (mean, median, mode, standard deviation, etc.) for each variable as a well-formatted table

NOTE: The report should be in {user_language}
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
Please provide a comprehensive data analysis for the following dataset by following the commend below:

{user_input_file}

Identification of any potential outliers or missing values
Report variables that has strong correlations and what are their significance(Avoid duplicated sentences)
Identify trends, patterns, correlations, or anomalies in the data and describe their significance.

And the report should be in {user_language}
Avoid duplicated sentences

The format should be (as an example):
topic(change topic to actual topic):
- point1(change point to actual outcome):
- point2(change point to actual outcome):
- point3(change point to actual outcome): 
...

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
Please provide a comprehensive data analysis for the following dataset by following the commend below:
{user_input_file}
Identify and describe the significance of trends, patterns, correlations, or anomalies in the data.
the report should be in {user_language}

The format should be (as an example):
topic(change topic to actual topic):
- point1(change point to actual outcome):
- point2(change point to actual outcome):
- point3(change point to actual outcome): 
...

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
Please provide a comprehensive data analysis for the following dataset by following the commend below:

{user_input_file}

Analyze {features_x} and {features_y} using {ml_model} model
Build the {ml_model} model between {features_x} and {features_y}
Calculate and display the equation of the {ml_model} model built above and explain its outcome parameters
Explain this model's meaning(Do not show the code)

And the report should be in {user_language}
Avoid duplicated sentences

...

Please present the findings in a clear, precise, and professional manner.
'''

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=0.5,
)
print(f'''{response_text.choices[0].text.strip()}''')

path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
