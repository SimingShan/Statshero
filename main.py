# Import all necessary packages
import openai
import runpy
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cleaning
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tabulate import tabulate
from statsmodels.tsa.stattools import adfuller

openai.api_key = "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"
path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path2 = "C:/Users/int_shansiming/Desktop/Prediction/data.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"

# ------------------------------------------
# Set up the parameters for the GPT-3 API
model_text = "text-davinci-002"
model_code = "text-davinci-002"
temperature_1 = 0.1
temperature_2 = 1
max_tokens = 3200

# ------------------------------------------
# Ask for file location
user_input_file = input("Enter the file location:");

# import the data
try:
    user_data = cleaning.clean(user_input_file)
except ValueError:
    user_data = cleaning.clean(user_input_file)

# Then get the column names
col_name = user_data.columns.tolist()

from request_matching import parse_user_input, select_variables_and_model

while True:
    # Enter request
    user_request = input("Please enter your request: ")

    # Get the desired variables from the input request
    parsed_msg = parse_user_input(user_input_file, user_request)
    # selected_v = select_variables_and_model(user_input_file, user_request)
    selected_v = []
    for v in col_name:
        if v in parsed_msg:
            selected_v.append(v)

    # Check if the selected_v set has more than one element
    if len(selected_v) > 1:
        break
    else:
        print("Please provide a request with at least two variables.")
print(col_name)
selected_type = []
for v in selected_v:
    v_type = user_data[v].dtype
    selected_type.append(v_type)

features_y = input(f"Select your y from {selected_v}: ");
y_index = selected_v.index(features_y)
selected_v.pop(y_index)
y_type = selected_type[y_index]
selected_type.pop(y_index)

prompt = f'''Based on the name and the type of x and y, and the request {parsed_msg}
        ,distinguish what is the best statistical model for user's request
        the name of x is {selected_v}
        the name of y is {features_y}
        the type of x is {selected_type} accordingly
        the type of y is {y_type}
        choose one suitable model based on bias-variance trade-off and based on variables'type
        for exmaple, when there are type of datetime in the {selected_type}, then a regression model should not be recommended
        you must only print the name of statistical model, no explainations needed
        '''
response_text = openai.Completion.create(
    engine=model_text,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=0.7,
)
rec_ml = f'''{response_text.choices[0].text.strip()}'''

# ------------------------------------------

print(" ")
print("\033[1m\033[4m\033[36mPlot\033[0m")
print(" ")

prompt = f'''
   Generate Python code to accomplish the following tasks:
1. Import cleaning.py and use the cleaning.clean({user_input_file}), save as 'df'.
2. Import matplotlib.pyplot as plt and create a plot to display the relationship between x = {selected_v} and y = {features_y}.
3. make the size of the plot: plt.figure(figsize=(12, 6))
4. Add a title to the graph using the Matplotlib library.
5. Label the axes using appropriate units based on the names of the features.
6. use plt.show() in the end of the code


Please provide the code without any additional comments or notes.
    '''
# Generate code using the GPT-3 API
response = openai.Completion.create(
    engine=model_code,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature_1,
)

# Save the generated code to a file
with open("generated_code.py", "w") as f:
    f.write(response.choices[0].text.strip())

# Import the generated code as a module
import generated_code

runpy.run_path("generated_code.py")

# ------------------------------------------
# Using openai api to generate a comprehensive report
print(" ")
print("\033[1m\033[4m\033[36mIntroduction to the variables\033[0m")
print(" ")
prompt_text = f'''
Using the Excel file {user_input_file}, please provide a brief introduction to the following variables: 
{col_name}. 
Please only describe these variables and do not create any new ones. 
'''

response_text = openai.Completion.create(
    engine=model_text,
    prompt=prompt_text,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

# --------------------------
# Generate the descriptive statistics
print(" ")
print("\033[1m\033[4m\033[36mDescriptive statistics\033[0m")
print(" ")
from Descriptive_statistics import des_chart

des_stats = des_chart(user_input_file)
print(tabulate(des_stats, headers='keys', tablefmt='github', showindex=True))
print(" ")
# Write a prompt that can read the descriptive statistics
prompt_des_stats = f'''
Here is a chart of descriptive statistics from the Excel file {user_input_file}:
{des_stats}
Please provide a detailed description and insights of the main characteristics and patterns in this summary chart
In a professional statistician's tongue.
'''

response_text = openai.Completion.create(
    engine=model_text,
    prompt=prompt_des_stats,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

# --------------------------
# Generate the correlation relationship chart
print(" ")
print("\033[1m\033[4m\033[36mCorrelation\033[0m")
print(" ")
from Descriptive_statistics import cor_chart

cor_stats = cor_chart(user_input_file)
print(tabulate(cor_stats, headers='keys', tablefmt='github', showindex=True))
print(" ")
prompt_cor_stats = f'''
Here is a chart of correlation from the Excel file {user_input_file}:
{cor_stats}
Please provide a detailed description and insights of 
the main characteristics and patterns in this correlation chart,
then analyze its insights
In a professional statistician's tongue.
'''

response_text = openai.Completion.create(
    engine=model_text,
    prompt=prompt_cor_stats,
    max_tokens=max_tokens,
    temperature=temperature_2,
)
print(f'''{response_text.choices[0].text.strip()}''')

term_mappings = {
    "time series analysis": ["time series analysis", "time series", "ARIMA", "SARIMAX", "seasonal decomposition",
                             "Holt-Winters"],
    "linear regression": ["linear regression", "LinearRegression", "OLS", "ordinary least squares", "linear model"],
    "logistic regression": ["logistic regression", "LogisticRegression", "logit", "logistic model"],
    "correlation analysis": ["correlation analysis", "correlation", "correlation coefficient", "Pearson", "Spearman",
                             "Kendall"],
    "support vector machines": ["support vector machines", "SVM", "support vector machine"],
    "decision trees": ["decision trees", "DecisionTree", "decision tree", "CART", "classification and regression tree"],
    "random forests": ["random forests", "RandomForest", "random forest"],
    "gradient boosting": ["gradient boosting", "GradientBoosting", "GBM", "XGBoost", "LightGBM", "CatBoost"],
    "neural networks": ["neural networks", "neural network", "deep learning", "artificial neural network", "ANN"],
    "k-means clustering": ["k-means clustering", "KMeans", "k-means"],
    "principal component analysis": ["principal component analysis", "PCA", "principal components"]}


def standardize_term(user_request, rec_ml):
    # First, search for a match in user_request
    for standard_term, synonyms in term_mappings.items():
        for syn in synonyms:
            if re.search(rf'\b{re.escape(syn.lower())}\b', user_request.lower()):
                return standard_term

    # If no match is found in user_request, search for a match in rec_ml
    for standard_term, synonyms in term_mappings.items():
        for syn in synonyms:
            if re.search(rf'\b{re.escape(syn.lower())}\b', rec_ml.lower()):
                return standard_term

    return None  # If no match is found, return None


standard_method_name = standardize_term(user_request, rec_ml)

###
# --------------------------
# --------------------------
# -------Time Series--------
# --------------------------
# --------------------------
# If the data is time series, then whether stationary?
if standard_method_name == "time series analysis":
    print(" ")
    print("\033[1m\033[4m\033[36mTime Series Analysis\033[0m")
    print(" ")

    from ADF_test import is_stationary
    from stationary_plot import sta_plt

    # Generate rolling mean and sd plot
    sta_plt(user_input_file, features_y)

    # Generate if stationary True/False
    message_st = is_stationary(user_input_file, features_y)

    prompt_cor_stats = f'''
    Here is a message about whether the Excel file {user_input_file} is stationary about feature {features_y}:
    {message_st}
    Please provide a detailed description about stationary,and analyze the meaning of the above message in context
    in a neat and professional statistician's tongue.
    The output should be in fowllowing format:
    A stationary time series data means that:
    The above graph indicates that the time series is:
    '''
    response_text = openai.Completion.create(
        engine=model_text,
        prompt=prompt_cor_stats,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(f'''{response_text.choices[0].text.strip()}''')

    # Perform ARIMA
    from arima import arima_model

    arima_model, arima_pred, arima_residuals, arima_summary = arima_model(user_input_file, features_y)
    # Use AI to explain the model
    prompt_arima = f'''Given an {arima_model} model
    please explain the model in detail. Assume that the model has already been fit to a time series dataset.
    Please explain the meaning and significance of each of these parameters 
    Additionally, please explain how the model was fit to the data, 
    and how the predict() and predict_in_sample() methods can be used to forecast future values of the time series.
    Finally, please provide any additional insights or observations about the model that you feel would be helpful in understanding its behavior and performance.
    '''
    response_text = openai.Completion.create(
        engine=model_code,
        prompt=prompt_arima,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(f'''{response_text.choices[0].text.strip()}''')

    prompt_arima_2 = f'''Given a list of residuals from {arima_residuals}, explain how well an ARIMA model has performed:

    The ARIMA model was used to forecast a time series. The model generated a list of residuals by subtracting the predicted values from the actual values. Please analyze the list of residuals and provide a detailed report on how well the ARIMA model has performed. Specifically, please answer the following questions:

    1. What is the mean of the residuals?
    2. What is the standard deviation of the residuals?
    3. Are the residuals normally distributed? 
    4. Is there any evidence of autocorrelation in the residuals? 
    5. Is there any evidence of heteroscedasticity in the residuals? 
    '''
    response_text = openai.Completion.create(
        engine=model_text,
        prompt=prompt_arima_2,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(f'''{response_text.choices[0].text.strip()}''')

    print(arima_summary)
    prompt_arima_2 = f'''Please explain the results table for a SARIMAX model generated using the auto_arima function in Python. 
    The table is displayed in the following format:
    {arima_summary}
    Please provide a detailed explanation of the table, including what each column and row represents, 
    and the significance of the coefficients and test statistics. 
    Based on the value of each coefficients and test statistics, explain how well the model is performing
    Additionally, please explain any other relevant information,
    such as the model order and how it was determined, and any assumptions that were made in fitting the model.
    '''
    response_text = openai.Completion.create(
        engine=model_text,
        prompt=prompt_arima_2,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(f'''{response_text.choices[0].text.strip()}''')
    # --------------------------
    # --------------------------
    # --------------------------
    # --------------------------
elif standard_method_name == "linear regression":
    print(" ")
    print("\033[1m\033[4m\033[36mLinear Regression\033[0m")
    print(" ")
    # --------------------------
    # --------------------------
    # ----linear regression-----
    # --------------------------
    # --------------------------
    # Load your dataset
    # Set the target variable and feature variables
    # Replace 'target_variable' with the column name of the target variable in your dataset
    # Replace 'feature1', 'feature2', ... with the column names of the feature variables in your dataset
    X = user_data[selected_v]
    y = user_data[features_y]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions using the test set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate residuals
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    # Calculate performance metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate and visualize residuals
    residuals = y_test - y_test_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.title("Test Residuals vs. True Values")
    plt.xlabel("True Values")
    plt.ylabel("Residuals")
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()

    # Scatter plot of true values vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.title("True Values vs. Test Predicted Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

    coef = model.coef_
    Inter = model.intercept_

    # Print essential information
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Training MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("Training R-squared:", train_r2)
    print("Test R-squared:", test_r2)

    prompt = f'''Please analyze the given list of residuals {train_residuals} of training group,
    and residuals {test_residuals} from the test group,and identify any trends or patterns that may be present. 
    Based on the observed trends, determine how the linear regression model is functioning and suggest possible improvements.
    '''
    response_text = openai.Completion.create(
        engine=model_text,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(" ")
    print("\033[1m\033[4m\033[36mSummary\033[0m")
    print(" ")
    print(f'''{response_text.choices[0].text.strip()}''')

    prompt = f'''Given the summarized features and performance metrics of a linear regression model,
    please analyze the model's performance. Identify any potential issues, such as multicollinearity,
    heteroscedasticity, or overfitting. Discuss the model's effectiveness in explaining the variance 
    in the target variable and suggest possible improvements or alternative modeling approaches.
    Coefficients: {coef}
    Intercept: {Inter}
    Training MSE: {train_mse}
    Test MSE:{test_mse}
    Training R-squared:{train_r2}
    Test R-squared: {test_r2}

    '''
    response_text = openai.Completion.create(
        engine=model_text,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(" ")
    print("\033[1m\033[4m\033[36mResidual\033[0m")
    print(" ")
    print(f'''{response_text.choices[0].text.strip()}''')

elif standard_method_name == "correlation analysis":
    print(" ")
    print("\033[1m\033[4m\033[36mCorrelation Analysis\033[0m")
    print(" ")
    # --------------------------
    # --------------------------
    # --Correlation Analysis----
    # --------------------------
    # --------------------------
    # Calculate correlation coefficients between the predictor and independent variables
    correlations = user_data[[features_y] + selected_v].corr()
    print(" ")
    print("\033[1m\033[4m\033[36mCoorelation Matrix\033[0m")
    print(" ")
    # Display the correlation matrix
    print("Correlation Matrix:")
    print(correlations)
    print(" ")
    print("\033[1m\033[4m\033[36mHeat Map\033[0m")
    print(" ")
    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()
    prompt = f'''Given the following correlation coefficients between a predictor variable {features_y}
    and several independent variables in the {correlations}, please provide a detailed interpretation of the relationships, 
    their strengths, and directions:
    Correlation between features_y and independent_var1: [value1]
    Correlation between features_y and independent_var2: [value2]
    Correlation between features_y and independent_var3: [value3]
    Explain the significance and meaning of these correlation coefficients, 
    as well as any potential implications for the relationship between the predictor and the independent variables.

    '''
    response_text = openai.Completion.create(
        engine=model_text,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature_2,
    )
    print(" ")
    print("\033[1m\033[4m\033[36mSummary\033[0m")
    print(" ")
    print(f'''{response_text.choices[0].text.strip()}''')
else:
    pass

print(standard_method_name)

path = "C:/Users/int_shansiming/Desktop/Prediction/Nasdaq.csv"
path3 = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
# analyze the relationshp between humidity wind speed meantemp and meanpressure using correlation analysis
