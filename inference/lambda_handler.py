import base64
import json

from HTML_page_model import HTML_PAGE
from online_inference import predict


def lambda_handler(event, context):

    method = (
        event
        .get("requestContext", {})
        .get("http", {})
        .get("method", "GET")
    )

    if method == "GET":

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/html; charset=utf-8"
            },
            "body": HTML_PAGE
        }

    if method == "POST":

        try:

            body = event.get(
                "body",
                ""
            )

            if event.get(
                "isBase64Encoded",
                False
            ):

                body = base64.b64decode(
                    body
                ).decode("utf-8")

            payload = json.loads(
                body
            )

            image_bytes = base64.b64decode(
                payload["image"]
            )

            predictions = predict(
                image_bytes
            )

            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type":
                        "application/json"
                },
                "body": json.dumps({
                    "predictions":
                        predictions
                })
            }

        except Exception as error:

            print(
                "Inference error:",
                repr(error)
            )

            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type":
                        "application/json"
                },
                "body": json.dumps({
                    "error":
                        str(error)
                })
            }

    return {
        "statusCode": 405,
        "body": "Method not allowed"
    }