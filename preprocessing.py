from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_creatinine_results(data):
    creatinine_columns = data.iloc[:,3:].copy()
    creatinine_results = []

    for _, row in creatinine_columns.iterrows():
        # Combine adjacent columns:
        row_combined = list(zip(row[::2], row[1::2]))
        row_combined = [(datetime.strptime(t[0], '%Y-%m-%d %H:%M:%S'), t[1])
                        for t in row_combined if
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
        time_since_previous_result
        past_48h_lowest_value
        past_48h_variance
        past_week_lowest_value
        past_year_median

    :param data: dataframe containing patient data
    :return: dataframe containing features
    """

    features = pd.DataFrame(
        columns=['age', 'sex', 'latest_result', 'previous_result', 'time_since_previous_result',
                 'past_48h_lowest_value', 'past_48h_variance', 'past_week_lowest_value', 'past_year_median'])

    features['age'] = data['age']

    le = LabelEncoder()
    features['sex'] = le.fit_transform(data['sex'])

    # [(time1, result1), (time2, result2), ...]:
    creatinine_results = get_creatinine_results(data)

    features['latest_result'] = pd.Series(creatinine_results).apply(
        lambda x: x[-1][1])

    # If there is no previous result, duplicate most recent result
    features['previous_result'] = pd.Series(creatinine_results).apply(
        lambda x: x[-2][1] if len(x) > 1 else x[-1][1])

    features['time_since_previous_result'] = pd.Series(
        creatinine_results).apply(
        lambda x: (x[-1][0] - x[-2][0]).total_seconds() if len(x) > 1 else 0)

    features['past_48h_lowest_value'] = pd.Series(creatinine_results).apply(
        lambda x: min(
            [y[1] for y in x if x[-1][0] - y[0] < timedelta(hours=48)]))

    features['past_48h_variance'] = pd.Series(creatinine_results).apply(
        lambda x: np.var(
            [y[1] for y in x if x[-1][0] - y[0] < timedelta(hours=48)]))

    features['past_week_lowest_value'] = pd.Series(creatinine_results).apply(
        lambda x: min(
            [y[1] for y in x if x[-1][0] - y[0] < timedelta(weeks=1)]))

    features['past_year_median'] = pd.Series(creatinine_results).apply(
        lambda x: np.median(
            [y[1] for y in x if x[-1][0] - y[0] < timedelta(weeks=52)]))

    return features