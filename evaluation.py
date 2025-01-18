import numpy as np

def calculate_precision(predictions, labels):
    """
    Calculate the precision from predictions and labels.

    Parameters:
        predictions (numpy.ndarray): Binary predictions array (0 or 1).
        labels (numpy.ndarray): Binary ground truth labels array (0 or 1).

    Returns:
        float: Precision value.
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Validate input shapes
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Calculate true positives (TP) and false positives (FP)
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))

    # Compute precision
    if true_positives + false_positives == 0:
        return 0.0  # Avoid division by zero

    precision = true_positives / (true_positives + false_positives)
    return precision


def calculate_recall(predictions, labels):
    """
    Calculate the recall from predictions and labels.

    Parameters:
        predictions (numpy.ndarray): Binary predictions array (0 or 1).
        labels (numpy.ndarray): Binary ground truth labels array (0 or 1).

    Returns:
        float: Recall value.
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # Validate input shapes
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape.")

    # Calculate true positives (TP) and false negatives (FN)
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_negatives = np.sum((predictions == 0) & (labels == 1))

    # Compute recall
    if true_positives + false_negatives == 0:
        return 0.0  # Avoid division by zero

    recall = true_positives / (true_positives + false_negatives)
    return recall

def f3_score(precision, recall):
    """
    Calculate the F3 score, which gives more weight to recall than precision.

    Parameters:
        precision (float): The precision value (between 0 and 1).
        recall (float): The recall value (between 0 and 1).

    Returns:
        float: The F3 score.
    """
    if precision == 0 and recall == 0:
        return 0.0  # Avoid division by zero

    beta = 3
    beta_squared = beta ** 2
    f3 = (1 + beta_squared) * (precision * recall) / ((beta_squared * precision) + recall)
    return f3

def calculate_f3_score(predictions, labels):
    """
    Calculate the F3 score given predictions and labels.
    :param predictions: numpy array of predictions.
    :param labels: numpy array of labels.
    :return: f3 score.
    """
    return f3_score(calculate_precision(predictions, labels),
                    calculate_recall(predictions, labels))