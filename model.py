#!/usr/bin/env python3

import argparse
import csv

import pandas as pd

from evaluation import calculate_f3_score
from preprocessing import extract_features
from training import training
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    model = training()
    test_data = pd.read_csv(flags.input)
    test_features = extract_features(test_data)
    predictions = model.predict(test_features)
    w = csv.writer(open(flags.output, "w"))
    w.writerow(("aki",))
    for prediction in predictions:
        w.writerow("y" if prediction else "n")

    if 'aki' in test_data.columns:
        le = LabelEncoder()
        print(calculate_f3_score(predictions, le.fit_transform(
            test_data['aki'].to_numpy())))


if __name__ == "__main__":
    main()