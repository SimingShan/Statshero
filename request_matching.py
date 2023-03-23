from fuzzywuzzy import fuzz, process
import openai
import cleaning
from cleaning import clean

openai.api_key = "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"


def parse_user_input(user_input_file, user_request):

    # import the data
    try:
        user_data = cleaning.clean(user_input_file)
    except ValueError:
        user_data = cleaning.clean(user_input_file)
    # Then get the column names
    col_name = user_data.columns.tolist()
    prompt = f'''
    Given a user input requesting to analyze the relationship between variables, and a dataset containing specific variables,
    please identify and match the variables mentioned in the user input with their corresponding actual variables in the dataset. 
    The user input is: {user_request}

    The actual variables in the dataset are: {col_name}. (Don't print this part)

    Please identify the variables from the user input and match them with the actual variables in the dataset.

    Rephrase the user's request with the correct variable names'''

    response = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )
    parsed_info = response.choices[0].text.strip()
    return parsed_info

def select_variables_and_model(user_input_file, parsed_info):
    # Variables
    # import the data
    try:
        user_data = cleaning.clean(user_input_file)
    except ValueError:
        user_data = cleaning.clean(user_input_file)
    # Then get the column names
    col_name = user_data.columns.tolist()
    available_variables = col_name
    selected_variables = []

    for variable in available_variables:
        # Find the best matches for the variable in the parsed info
        matches = process.extract(variable.lower(), parsed_info.lower().split())

        # Filter the matches based on a higher threshold (e.g., 95)
        filtered_matches = [match for match, score in matches if score > 85]

        if filtered_matches:
            # Consider the highest-scoring match a valid match
            selected_variables.append(variable)

    return selected_variables

