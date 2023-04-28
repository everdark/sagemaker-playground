"""User module for model inference to be used in SageMaker script mode."""
import json
import os
from enum import Enum
from io import BytesIO, StringIO
from typing import BinaryIO

import joblib
import pandas as pd
from botocore.response import StreamingBody
from sklearn.base import BaseEstimator


class ContentType(str, Enum):
    json = "application/json"
    jsonline = "application/jsonlines"
    csv = "text/csv"
    parquet = "application/x-parquet"


def model_fn(model_dir: str) -> BaseEstimator:
    """Model function to load a pre-trained scikit-learn model.

    Args:
        model_dir: Path to the model artifacts. By default it is /opt/ml/model/.

    Returns
        A pre-trained model object.

    """
    # NOTE: in local mode, the entire directory in which the module calling .deploy() is
    #       is copied to model_dir
    model = joblib.load(os.path.join(model_dir, "models/model.joblib"))
    return model


def predict_fn(data: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    """Prediction function."""
    yhat = model.predict(data)
    out = pd.DataFrame(yhat, columns=["predicted_score"], index=data.index)
    return out


def input_fn(input_data: StreamingBody, content_type: ContentType) -> pd.DataFrame:
    """Input function for processing incoming data before model prediction."""
    label_name = "species"
    if content_type == ContentType.parquet:
        data = BytesIO(input_data)
        df = pd.read_parquet(data)
    elif content_type == ContentType.csv:
        df = pd.read_csv(StringIO(input_data))
    elif content_type == ContentType.json:
        df = pd.json_normalize(json.loads(input_data))
    else:
        raise ValueError(
            f"content type should be one of: {[c.value for c in ContentType]}"
        )

    if label_name in df.columns:
        df = df.drop(columns=[label_name])

    print(df)

    return df


def output_fn(output: pd.DataFrame, accept: ContentType) -> BinaryIO:
    """Output function for post-processing model prediction output."""
    if accept == ContentType.parquet:
        buffer = BytesIO()
        output.to_parquet(buffer)
        return buffer.getvalue()
    elif accept in [ContentType.json, ContentType.jsonline]:
        return json.dumps(output.to_dict(orient="records"))
    else:
        raise Exception("Requested unsupported ContentType in Accept: " + accept)
