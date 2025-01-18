import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from preprocessing import extract_features


def training():
    training_data = pd.read_csv('training.csv')
    x = extract_features(training_data) # x & y are pandas dataframes
    le = LabelEncoder()
    y = le.fit_transform(training_data.loc[:,"aki"])

    model = XGBClassifier(objective='binary:logistic', random_state=420)
    model.fit(x, y)

    return model