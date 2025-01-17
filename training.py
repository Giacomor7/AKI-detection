import pandas as pd

from preprocessing import extract_features


def training():
    training_data = pd.read_csv('training.csv')
    x = extract_features(training_data) # x & y are pandas dataframes
    y = training_data.loc[:,"aki"]