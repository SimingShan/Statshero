# Set up the parameters for the GPT-3 API
import openai
import runpy
import subprocess
import matplotlib.pyplot as plt
openai.api_key = "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"

# Ask user for input
user_input_1 = input("Enter the prompt")

# Ask for features if the user ask for a plot
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    features_x = input("Enter the x")
    features_y = input("Enter the y")
else:
    user_input_1 = user_input_1

# Ask for file location
user_input_file = input("Enter the file location:")

# Check if the input contains any keywords
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    prompt = "Generate Python code that imports the dataset from " \
             + user_input_file \
             + ", Generate Python code that cleans the dataset" \
             + ", Using pd.to_numeric() to change all numbers' type to float" \
             + ", Using pd.to_datetime() to change all date like columns type to datetime" \
             + ", then creates and print a table that summary the dataset features"\
             + ", Generate Python code that plot the relationship of x = " \
             + features_x \
             + "and y = "\
             + features_y \
             + " using the Matplotlib library" \
             + "Following the above prompt strictly! No annotation to the code!"
else:
    prompt = user_input_1 + ", The file is from: " + user_input_file

model = "text-davinci-002"
temperature = 0.4
max_tokens = 500

# Generate code using the GPT-3 API
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature,
)

# Save the generated code to a file
with open("generated_code.py", "w") as f:
    f.write(response.choices[0].text.strip())

# Import the generated code as a module

import generated_code

runpy.run_path('generated_code.py')

prompt_text = "Generate a report on the dataset from " \
            + user_input_file \
            + "The report should include a summary of the dataset, any notable trends or patterns in the data, " \
              "key insights that can be gleaned from the data, and any other relevant information. " \
              "Please use clear and concise language, and include visualizations if appropriate."

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=temperature,
)
print(response_text.choices[0].text.strip())
