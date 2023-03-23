import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse

def find_unnecessary_columns(df, missing_threshold=0.5, variance_threshold=0.1):
    unnecessary_columns = []
    # Find columns with a high proportion of missing values
    missing_ratios = df.isnull().sum() / len(df)
    high_missing_columns = missing_ratios[missing_ratios > missing_threshold].index.tolist()
    unnecessary_columns.extend(high_missing_columns)

    # Find columns with low variance (for numerical columns)
    low_variance_columns = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).var() < variance_threshold].tolist()
    unnecessary_columns.extend(low_variance_columns)
    return unnecessary_columns

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
    # Drop unnecessary columns
    un_col = find_unnecessary_columns(data)
    data = data.drop(columns=un_col, errors='ignore')
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

    data.columns = data.columns.str.replace('_', ' ')
    return data

