#!/usr/bin/env python
"""Train a simple scikit-learn model (without SageMaker) using the iris dataset."""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

_INPUT_DATA_PATH = "data/iris.csv"
_OUTPUT_MODEL_PATH = "models/model.joblib"


if __name__ == "__main__":
    df = pd.read_csv(_INPUT_DATA_PATH)
    model = LogisticRegression(C=1.23456, max_iter=987, random_state=42)
    model.fit(df.iloc[:, :4], df["species"])
    joblib.dump(model, _OUTPUT_MODEL_PATH)
