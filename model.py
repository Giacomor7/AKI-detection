#!/usr/bin/env python3

import argparse
import csv

import pandas as pd

from preprocessing import extract_features
from training import training


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


if __name__ == "__main__":
    main()