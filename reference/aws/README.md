# AWS reference deployment

An alternative serving path for the same model on AWS, built during the AWS AI & ML Scholars program labs:

```
web/index.html  ->  API Gateway  ->  lambda/lambda_function.py  ->  SageMaker XGBoost endpoint
                                                                   (notebooks/train_and_deploy.ipynb)
```

- `notebooks/train_and_deploy.ipynb` trains the built-in SageMaker XGBoost algorithm on `data/processed/features_2011_09_01.csv` and deploys a serverless endpoint.
- `lambda/lambda_function.py` validates the ten features, calls the endpoint and returns the probability.
- `web/index.html` is a static form that calls the API.

This path needs an AWS account and is not part of CI. The maintained deployment is the Docker image in the repository root (Oracle Cloud Always Free, with Hugging Face Spaces as a backup). Delete the endpoint after a demo to avoid charges.
