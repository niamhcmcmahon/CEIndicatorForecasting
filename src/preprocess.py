import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05  # p < 0.05 → stationary

def difference_series(series):
    return series.diff().dropna()

def train_test_split(df, split_ratio=0.8):
    n = int(len(df) * split_ratio)
    train = df.iloc[:n]
    test = df.iloc[n:]
    return train, test
