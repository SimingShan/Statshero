from rouge import Rouge
import openai
import cleaning
import numpy as np
from cleaning import clean
from Descriptive_statistics import des_chart

path = "C:/Users/int_shansiming/Desktop/Prediction/DailyDelhiClimateTest.csv"
openai.api_key = "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"
example = '''
The average daily temperature during this period was 19.97°C, with a moderate variability as indicated by a standard 
deviation of 4.93°C.
The temperature seems to increase over time, as observed by the higher temperatures in April compared to January. 
This could be due to seasonal changes.
The humidity levels show a slightly lower mean value (62.22%) compared to the median (67.21%), 
which indicates the distribution may be slightly skewed towards lower humidity levels.
Wind_speed shows a wide range of values, with a standard deviation of 3.76 m/s, 
indicating that the wind conditions can be quite variable during this period.
The mean pressure value has an anomalous minimum value of 59 hPa, which appears to be an error. 
This value affects the standard deviation, making it larger than expected.
In conclusion, the dataset provides an overview of the weather patterns for the given period, 
with temperatures showing an increasing trend, possibly due to seasonal changes. 
Humidity levels and wind speeds show significant variability during this period, 
while the pressure data should be carefully evaluated due to the presence of an anomalous value.'''
summary_table = des_chart(path)
best_prompt = None
best_score = None
prompt_candidate = []

example_p = ''' Here is a chart of descriptive statistics from the Excel file {}
    Please provide a detailed description and insights of the main characteristics and patterns in this summary chart
    In a professional statistician's tongue'''

def gen_prompt(example_p, number):
    prompt = '''Please suggest different ways to phrase the following request remember to include placeholder of the excel file: {} '''

    for _ in range(number):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt.format(example_p),
            max_tokens=len(example_p)*number + len(prompt)*5,
            n=1,
            stop=None,
            temperature=0.8,
        )
        generated_summary = response.choices[0].text.strip()
        prompt_candidate.append(generated_summary)

    return prompt_candidate

prompt_candidate = gen_prompt(example_p, 10)
prompt_candidate.append(example_p)

for i in range(len(prompt_candidate)):
    if "{}" not in prompt_candidate[i]:
        prompt_candidate[i] = prompt_candidate[i] + "{}"
    else:
        pass



def get_rouge(prompt, example):
    rouge = Rouge()
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt.format(summary_table),
        max_tokens=3000,
        n=1,
        stop=None,
        temperature=0.7,
    )
    generated_summary = response.choices[0].text.strip()
    scores = rouge.get_scores(generated_summary, example)
    # Use ROUGE-L F1 score for comparison
    rouge_l_f1_score = scores[0]["rouge-l"]["f"]
    return rouge_l_f1_score, generated_summary


score_lst = [[] for _ in range(len(prompt_candidate))]
response_lst = [[] for _ in range(len(prompt_candidate))]

for candidate_idx, prompt in enumerate(prompt_candidate):

    for i in range(10):
        prompt_score, gen_response = get_rouge(prompt, example)
        score_lst[candidate_idx].append(prompt_score)
        response_lst[candidate_idx].append(gen_response)



score_lst_mean = []

for i in range(len(score_lst)):
    score_lst_mean.append(np.mean((score_lst[i])))

best_index = score_lst_mean.index(max(score_lst_mean))

print("The best prompt is: " + prompt_candidate[best_index])
print(" ")
for i in range(len(response_lst[best_index])):
    print(response_lst[best_index][i])


