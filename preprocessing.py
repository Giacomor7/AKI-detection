import pandas as pd

def get_latest_result(data):
    """
    Get value and
    :param data:
    :return:
    """

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
    features['latest_result'] = # rightmost non-null column for each row...