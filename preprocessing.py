import numpy as np
import pandas as pd


def get_creatinine_results(data):
    creatinine_columns = data.iloc[:,3:].copy()
    creatinine_results = []

    for _, row in creatinine_columns.iterrows():
        # Combine adjacent columns:
        row_combined = list(zip(row[::2], row[1::2]))
        row_combined = [t for t in row_combined if
                        not pd.isna(t[0]) and not pd.isna(t[1])]
        creatinine_results.append(row_combined)

    return creatinine_results


def extract_features(data):
    """
    Features to extract (taken from NHS algorithm):
        age
        sex
        latest_result
        previous_result
        days_since_previous_result
        past_48h_lowest_value
        past_48h_variance
        past_week_lowest_value
        past_year_median

    :param data: dataframe containing patient data
    :return: dataframe containing features
    """

    features = pd.DataFrame(
        columns=['age', 'sex', 'latest_result', 'previous_result', 'days_since_previous_result',
                 'past_48h_lowest_value', 'past_48h_variance', 'past_week_lowest_value', 'past_year_median'])

    features['age'] = data['age']
    features['sex'] = data['sex']

    creatinine_results = get_creatinine_results(data)

    features['latest_result'] = pd.Series(creatinine_results) # use apply to get the latest result for each element