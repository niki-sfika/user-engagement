import pandas as pd
import numpy as np
from datetime import datetime, date
import copy
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.externals import joblib


"""
This script contains all the helper functions required 
"""

def show_me_basic_stats(df, cols_to_exclude=[], only_histograms=False):
    """
    Creates distribution plots and basic stat tables 
    :param df: pandas dataframe with the data
    :param cols_to_exclude: list of colums as strings to exclude from the analysis
    """
    
    df = df.drop(columns=cols_to_exclude)
    
    # Numerical features
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if len(df.select_dtypes(include=numerics).columns)>0:
        display(df.describe())
        hist = df.hist(bins=30, sharey=True, figsize=(10, 10))
    
    if not only_histograms:
        # categorical features
        for column in df.select_dtypes(include=['object']).columns:
            if column not in cols_to_exclude:
                display(pd.crosstab(index=df[column], columns='% observations', normalize='columns'))

                
def array_to_deciles(array):
    d1, d2, d3, d4, d5, d6, d7, d8, d9 = np.percentile(array, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    array = copy.deepcopy(array)
    for row in range(array.shape[0]):
        if 0.0 <= array[row] < d1:
            array[row] = 0.1
        elif d1 <= array[row] < d2:
            array[row] = 0.2
        elif d2 <= array[row] < d3:
            array[row] = 0.3
        elif d3 <= array[row] < d4:
            array[row] = 0.4
        elif d4 <= array[row] < d5:
            array[row] = 0.5
        elif d5 <= array[row] < d6:
            array[row] = 0.6
        elif d6 <= array[row] < d7:
            array[row] = 0.7
        elif d7 <= array[row] < d8:
            array[row] = 0.8
        elif d8 <= array[row] < d9:
            array[row] = 0.9
        else:
            array[row] = 1.0
    return array


def normalise_data(dataset):
    # get all numeric columns
    numeric_columns = list(dataset.select_dtypes(include="number").columns.values)
    scaler = MinMaxScaler()

    # apply minmax scaler
    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])
    return dataset


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))



