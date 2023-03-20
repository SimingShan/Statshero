import numpy as np
import pandas as pd
from tabulate import tabulate

# Write a function that generate the chart
# Read the Excel file
def des_chart(user_input_file):
    data = pd.read_csv(user_input_file)
    # Compute comprehensive descriptive statistics
    desc_stats = data.describe(include=[np.number]).transpose()
    # Drop unnecessary columns
    desc_stats = desc_stats.drop(columns=['count'])
    # Display the outcome in a well-formated table form
    return desc_stats


def cor_chart(user_input_file):
    data = pd.read_csv(user_input_file)
    cor_stats = data.corr(numeric_only=True)
    return cor_stats

