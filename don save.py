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

prompt_seed = """Generate a prompt that can create a similar summary from a descriptive statistics table. The example summary is: {}
Remember to write the structure, format, general content as the example summary,
Notice that the prompt should be general and not limited to the features mentioned in the example."""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt_seed.format(example),
    max_tokens=3000,
    n=1,
    stop=None,
    temperature=0.8,
)
example_p = response.choices[0].text.strip()

print(example_p)

prompt_seed = '''Please create a prompt that can generate the similar summary from a descriptive statistics， 
the example summary is : {}
Remember to write the desired structure, format, content and so on'''

















