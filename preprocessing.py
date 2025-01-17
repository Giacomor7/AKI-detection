import pandas as pd

def get_latest_result(data):
    """
    Get value and
    :param data:
    :return:
    """

def combine_result_and_time(data):
    """
    Combine each creatinine result and date into a single tuple
    :param data: patient data as dataframe
    :return: dataframe containing tuples for each result
    """
    creatinine_results = pd.DataFrame(
        columns=[f"result_{i}" for i in range(44)])

    for i in range(44):
        creatinine_results[f"result_{i}"] = (
        data[f"creatinine_result_{i}"], data[f"creatinine_date_{i}"])

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

    creatinine_results = combine_result_and_time(data)

    features = pd.DataFrame(
        columns=['age', 'sex', 'latest_result', 'previous_result', 'days_since_previous_result',
                 'past_48h_lowest_value', 'past_48h_variance', 'past_week_lowest_value', 'past_year_median'])

    features['age'] = data['age']
    features['sex'] = data['sex']
    features['latest_result'] = # rightmost non-null column for each row...