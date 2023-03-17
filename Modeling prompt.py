if any(keywords in ml_model for keywords in ["ARIMA", "time series"]):
    prompt_ml = f'''
    Generate Python code to accomplish the following tasks:
1. Import cleaning.py and use the cleaning.clean({user_input_file}), save as 'df'
2. Build a {ml_model} model with df upon df[{features_x}] and df[{features_y}]
3. Must import statsmodels.tsa.arima.model.ARIMA
4. Must Include necessary imports, data preprocessing, and model training steps.
Please provide the code only without any additional comments or notes or symbols '''
else:
    prompt_ml = f'''
    Generate Python code to accomplish the following tasks:
1. Import cleaning.py and use the cleaning.clean({user_input_file}), save as 'df'
2. Build a {ml_model} model with df upon df[{features_x}] and df[{features_y}]
3. In ARIMA(), do not use disp= argument
3. Must necessary imports, data preprocessing, and model training steps.
Please provide the code only without any additional comments or notes or symbols '''

response = openai.Completion.create(
    engine=model,
    prompt=prompt_ml,
    max_tokens=2000,
    n=1,
    stop=None,
    temperature=0.5, )

with open("ml_code.py", "w") as f:
    f.write(response.choices[0].text.strip())

# Import the generated code as a module
import ml_code

runpy.run_path('ml_code.py')
