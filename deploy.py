#!/usr/bin/env python
import os
import pathlib

import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.local import LocalSession
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn.model import SKLearnModel, SKLearnPredictor

_USER_MODULE_PATH = "mms_user_module.py"


def deploy_local_model(
    model_data_uri: str,
    entry_point: str,
    endpoint_name: str = "test_model",
    full_local_mode: bool = True,
) -> SKLearnPredictor:
    """Deploy a pre-trained model to SageMaker local endpoint.

    The endpoint will be running at the background at localhost:8080.
    Run "docker ps" to show the container hosting the served model.

    A valid AWS session is still required even though the model is not serving
    on AWS cloud.

    To test model inference from a terminal:
        curl --location '127.0.0.1:8080/invocations' \
        --header 'Content-Type: application/json' \
        --data '{
            "sepal_length": 1,
            "sepal_width": 1,
            "petal_length": 1,
            "petal_width": 1
        }'

    Args:
        model_data_uri: URI to a model package (.tar.gz), starting with either
            "s3://" or "file://".
        entry_point: Path to the SAGEMAKER_PROGRAM module script.
        endpoint_name: Name of the endpoint. This does not actually matter for a local
            deployment.

    Returns:
        A predictor.
        A container will be spawn and running as side effect.

    """
    iam_role = sagemaker.get_execution_role()
    model = SKLearnModel(
        model_data=model_data_uri,
        role=iam_role,
        # under the hood, this entrypoint module will be pacakged as sourcedir.tar.gz
        # and upload to a templated s3 path under the running aws account
        # if we are not running full local mode (local_code + local_mode)
        # the container will then download the package and install it as a standalone
        # python package under /opt/ml/code
        # this is refered to as the "script mode" in the aws docs
        # the script is also called "user module" in `sagemaker-scikit-learn-container`,
        # a python package which will be installed by default for a pre-built container
        # for more details:
        #   https://github.com/aws/sagemaker-scikit-learn-container
        entry_point=entry_point,
        # if we do full local mode, this arg is REQUIRED and all scripts under this
        # dir will be treated as module, under BOTH /opt/ml/code and /opt/ml/model
        # if we only do local_mode but not local_code, the directory where this calling
        # module is located will be mounted onto ONLY /opt/ml/model, and
        # /opt/ml/code will contain only the user module script
        source_dir="./code" if full_local_mode else None,
        image_uri=None,  # a default sklearn image will be downloaded (if not cached)
        framework_version="1.2-1",
    )

    # to make sure the session is fully local
    # without this, user module will be packaged and upload to s3 everytime
    # when deploy is called (refer to docs of the arg code_location for details)
    if full_local_mode:
        sagemaker_session = LocalSession()
        sagemaker_session.config = {"local": {"local_code": True}}
        model.sagemaker_session = sagemaker_session

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="local",  # a local sagemaker session will be automatically init
        endpoint_name=endpoint_name,
        # the serializer/deserializer will be convenient ONLY if we want to call
        # the .predict() method directly
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )
    return predictor


if __name__ == "__main__":
    # NOTE: this uri does not really matter in local mode for .deploy()
    #       the entire directory is mounted onto /opt/ml/model instead
    #       it probably matters wen we call the .register() method where the
    #       package will be located by sagemaker model registry
    model_data_uri = os.path.join(pathlib.Path().absolute().as_uri(), "model.tar.gz")

    predictor = deploy_local_model(
        model_data_uri=model_data_uri,
        entry_point=_USER_MODULE_PATH,
    )

    # to test the predictor in an interactive session...
    sample_data = {
        "sepal_length": 1,
        "sepal_width": 1,
        "petal_length": 1,
        "petal_width": 1,
    }
    print(f"prepare sample data for prediction: {sample_data}")

    response = predictor.predict(sample_data)
    print(f"sample prediction: {response[0]}")

    # predictor.delete_endpoint()  # gracefully shutdown the endpoint
