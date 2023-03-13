import openai
import runpy
import matplotlib.pyplot as plt
Filepath = "C:/Users/int_shansiming/Desktop/songs.xlsx"
OpenAI_API = "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"
model = "davinci:ft-personal-2023-03-10-02-45-08"
openai.api_key = "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"

try:
    # Set up the parameters for the GPT-3 API
    prompt = (
        "Generate a Python code that imports and cleans a dataset, "
        "Generate a Python code which can creates then print a table that summary the dataset features"
        "Generate a Python code which can display the plot of 'x-axis=Date','y-axis=High',"
        "The data file path is 'C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv'"
    )
    model = "text-davinci-002"
    temperature = 0.7
    max_tokens = 1000

    # Generate code using the GPT-3 API
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if
    # Save the generated code to a file
    with open("generated_code.py", "w") as f:
        f.write(response.choices[0].text.strip())

    # Import the generated code as a module
    import generated_code
    runpy.run_path('generated_code.py')
except:
    print("Sorry, there was an error, please try to reformulate your prompt. ")