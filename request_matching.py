from fuzzywuzzy import fuzz, process
import openai
import cleaning
from cleaning import clean

openai.api_key = 'your_api_key_here'
def parse_user_input(user_request):
    prompt = f'''Parse the following user input and identify the variables and try to provide solutions to the
             requirements: {user_request}'''

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
        filtered_matches = [match for match, score in matches if score > 90]

        if filtered_matches:
            # Consider the highest-scoring match a valid match
            selected_variables.append(variable)

    return selected_variables
