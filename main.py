# Import all necessary packages
import openai
import runpy
import subprocess
import matplotlib.pyplot as plt
openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"


# Set up the parameters for the GPT-3 API
model = "text-davinci-002"
temperature_1 = 0.1
temperature_2 = 0.7
max_tokens = 2000

# Ask user for input
user_input_1 = input("Enter your request")

# Ask for features if the user ask for a plot
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    features_x = input("Enter the x")
    features_y = input("Enter the y")
    method = input("Enter the plot type")
else:
    user_input_1 = user_input_1

# Ask for file location
user_input_file = input("Enter the file location:")

# Ask for whether needs cleaning
cleanornot = input("would you like to clean your dataset first?[yes or no]")

# if ask for cleaning
if cleanornot == 'yes':
    import cleaning
    cleaned_data = cleaning.clean(user_input_file)
    user_input_file = input("Enter your cleaned dataset location:")
    cleaned_data.to_csv(user_input_file)
else:
    pass

# Check if the input contains any keywords
if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    prompt = "Generate Python code that imports the dataset from " \
             + user_input_file \
             + ". Generate Python code that cleans the dataset" \
             + ", then creates and print a table that summary the data's numerical features"\
             + ", Generate Python code generate a " \
             + method \
             + " which displays the relationship between x = " \
             + features_x \
             + " and "\
             + features_y \
             + " add a legend(if needed) and a title to the graph"\
             + " using the Matplotlib library" \
             + " Following the above prompt strictly!" \
             + " No notes to the code!"


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
    prompt = user_input_1 + ", The file is from: " + user_input_file
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_1,
    )
    print(response.choices[0].text.strip())

prompt_text = "Generate a report on the dataset from " \
            + user_input_file \
            + "The report should include a summary of the dataset, any notable trends or patterns in the data, " \
              "key insights that can be gleaned from the data, and any other relevant information. " \
            + "Then, remember to analyze relationship between x = " \
            + features_x \
            + " and "\
            + features_y \
            + "Please use clear and concise language."

response_text = openai.Completion.create(
    engine=model,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print("Brief Summary Of the Data: " + response_text.choices[0].text.strip())
