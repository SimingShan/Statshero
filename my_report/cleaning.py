# Universal cleaning dataset ultimate code
import numpy as np
import pandas as pd

def clean(data):
    # import the dataset
    data = pd.read_csv(data)

    # Drop the missing data
    data = data.dropna()

    # Drop the duplicated data
    data = data.drop_duplicates()

    # Clean the inconsistent data type
    # Logic:
    # 1: Clean the float first, filter out all the float type
    # 2: Clean the date type, filter out all the date type
    # 3: All the remains are categorical

    # 1: Clean the float first, filter out all the float type
    float_cols = []
    for col in data.columns:
        try:
            data[col] = data[col].astype(float)
            float_cols.append(col)
        except ValueError:
            pass

    # 2: Clean the date type, filter out all the date type
    # 2(1): filter out all the non-float data
    non_float_cols = []
    for col in data.columns:
        if col in float_cols:
            pass
        else:
            non_float_cols.append(col)
    # 2(2):
    for col in non_float_cols:
        try:
            data[col] = pd.to_datetime(data[col], format='%m/%d/%Y')
        except ValueError:
            pass
    return data





