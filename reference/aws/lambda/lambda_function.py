"""
AWS Lambda: receives customer features from API Gateway, calls the
SageMaker endpoint, returns the repurchase probability.

Environment variable required: ENDPOINT_NAME
"""
import json
import os

import boto3

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "unique_products",
            "tenure_days", "return_rate", "orders_last_90d", "avg_days_between_orders", "is_uk"]
THRESHOLD = 0.5

runtime = boto3.client("sagemaker-runtime")

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}


def respond(status, body):
    return {"statusCode": status, "headers": CORS, "body": json.dumps(body)}


def lambda_handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return respond(200, {})

    try:
        body = event.get("body", event)
        data = json.loads(body) if isinstance(body, str) else body
        # one-order customers have no gap between orders; the training data encodes it as -1
        if data.get("avg_days_between_orders") is None:
            data["avg_days_between_orders"] = -1
        row = ",".join(str(float(data[f])) for f in FEATURES)
    except (KeyError, ValueError, TypeError) as e:
        return respond(400, {"error": f"Bad input: {e}", "required_fields": FEATURES})

    resp = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                   ContentType="text/csv", Body=row)
    prob = float(resp["Body"].read().decode("utf-8").strip().split(",")[0])

    return respond(200, {
        "repurchase_probability": round(prob, 4),
        "prediction": "likely to buy again" if prob >= THRESHOLD else "at risk",
        "recommended_action": ("include in loyalty campaign" if prob >= THRESHOLD
                               else "send win-back offer"),
    })
