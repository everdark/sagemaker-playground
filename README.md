# SageMaker Playground

The repository contains codes playing around with AWS SageMaker.

## Local Mode

Deploy a pre-trained scikit-learn model (NOT trained by SageMaker) to an local endpoint.

```bash
# train a model
./train-withou-sagemaker.py

# package the resulting model
make model-package

# deploy the model
./deploy
```

Now test model inference with `application/json` type:

```bash
curl --location '127.0.0.1:8080/invocations' \
--data '{
    "sepal_length": 1,
    "sepal_width": 1,
    "petal_length": 1,
    "petal_width": 1
}'
```

Or `text/csv` type:

```bash
curl --location '127.0.0.1:8080/invocations' \
--header 'Content-Type: text/csv' \
--data 'sepal_length,sepal_width,petal_length,petal_width
1,1,1,1'
```

Or `application/jsonlines`:

```bash
curl --location '127.0.0.1:8080/invocations' \
--header 'Content-Type: application/jsonlines' \
--data '{"sepal_length": 1, "sepal_width": 1, "petal_length": 1, "petal_width": 1}
{"sepal_length": 1, "sepal_width": 1, "petal_length": 1, "petal_width": 1}'
```
