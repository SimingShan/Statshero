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

######################################################################

if any(keyword in user_input_1 for keyword in ["plot", "graph", "analyze", "analysis"]):
    prompt = f'''Introduce {method} and explain how they are used in data analysis  in {user_language}.'''

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_1,
    )
