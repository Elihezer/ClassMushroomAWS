import io
import json
import os
import base64

import boto3
import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision.models import (
    ResNet18_Weights,
    resnet18
)

from HTML_page_model import HTML_PAGE


MODEL_BUCKET = os.environ.get(
    "MODEL_BUCKET",
    "mushroom-ml-mikael-2026-000"
)

MODEL_KEY = os.environ.get(
    "MODEL_KEY",
    "inference/model/v1/model.pt"
)

LABEL_MAPPING_KEY = os.environ.get(
    "LABEL_MAPPING_KEY",
    "inference/model/v1/label_mapping.json"
)


LOCAL_MODEL_PATH = "/tmp/model.pt"
LOCAL_LABEL_PATH = "/tmp/label_mapping.json"


s3 = boto3.client("s3")

s3.download_file(
    MODEL_BUCKET,
    MODEL_KEY,
    LOCAL_MODEL_PATH
)

s3.download_file(
    MODEL_BUCKET,
    LABEL_MAPPING_KEY,
    LOCAL_LABEL_PATH
)


with open(
    LOCAL_LABEL_PATH,
    "r",
    encoding="utf-8"
) as file:

    label_mapping = json.load(file)


id_to_name = {
    int(class_id): mushroom_name
    for mushroom_name, class_id
    in label_mapping.items()
}


weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=None
)

number_of_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.0),
    nn.Linear(
        number_of_features,
        len(label_mapping)
    )
)


model.load_state_dict(
    torch.load(
        LOCAL_MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

transform = weights.transforms()


def predict(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    image_tensor = (
        transform(image)
        .unsqueeze(0)
    )

    with torch.inference_mode():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    top_probabilities, top_classes = torch.topk(
        probabilities,
        k=3
    )

    predictions = []

    for probability, class_id in zip(
        top_probabilities,
        top_classes
    ):

        class_id = class_id.item()

        predictions.append({
            "name": id_to_name[class_id],
            "probability": round(
                probability.item() * 100,
                2
            )
        })

    return predictions

def lambda_handler(event, context):

    method = (
        event
        .get("requestContext", {})
        .get("http", {})
        .get("method", "GET")
    )

    # Browser asks for the web page.
    if method == "GET":

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/html; charset=utf-8"
            },
            "body": HTML_PAGE
        }

    # Browser sends an image for inference.
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
                ).decode(
                    "utf-8"
                )

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
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "predictions": predictions
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
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "error": str(error)
                })
            }

    return {
        "statusCode": 405,
        "body": "Method not allowed"
    }