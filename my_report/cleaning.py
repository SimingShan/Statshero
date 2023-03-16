import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse

def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False

def clean(file_path):
    # import the dataset
    try:
        data = pd.read_csv(file_path)
    except ValueError:
        data = pd.read_excel(file_path)
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
        if data[col].apply(is_date).all():
            data[col] = pd.to_datetime(data[col].apply(parse))
    return data

