user_input_file = 'C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv'
openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"

# ------------------------------------------
# Set up the parameters for the GPT-3 API
model = "text-davinci-002"
temperature_1 = 0.1
temperature_2 = 1
max_tokens = 3200
# --------------------------------------------
import importlib

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
print only the name of the best model, no explainations needed, the text should
'''
response_text = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=0.5,
)
rec_ml = response_text.choices[0].text.strip()
rec_ml_msg = "The recommended model for these features is " + f'''{response_text.choices[0].text.strip()}'''
print(rec_ml)

prompt = f'''
Write Python code that:
Create a fucntion named "gen_function(data, x, y)" using the {rec_ml} method using variables data[x] and data[y].
this function can also generates predictions using the fitted model,
this function can also generates plots the original data and the predictions using matplotlib.pyplot library.
Please exclude comments or notes from the code.
    '''
# Generate code using the GPT-3 API
response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature_1,
)
print(prompt)

# Save the generated code to a file
with open("gen_ml_code.py", "w") as f:
    f.write(response.choices[0].text.strip())

# Import the generated code as a module
try:
    import gen_ml_code

    importlib.reload(gen_ml_code)  # Reload the module to ensure the latest version is used
except ImportError:
    import gen_ml_code

# Import the generated code as a module
from gen_ml_code import gen_function

gen_function(user_data, features_x, features_y)