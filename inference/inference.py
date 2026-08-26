import sys
import json

import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
import requests
from io import BytesIO

import boto3


bucket_name = "mushroom-ml-mikael-2026-000"
model_key = "inference/model/v1/model.pt"
model_label_key = "inference/model/v1/label_mapping.json"
local_model_path = "model.pt"
local_label_path = "label_mapping.json"

s3_client = boto3.client("s3")

s3_client.download_file(
    bucket_name,
    model_key,
    local_model_path
)

s3_client.download_file(
    bucket_name,
    model_label_key,
    local_label_path
)



image_path = sys.argv[1]

if image_path.startswith("http://") or image_path.startswith("https://"):

    response = requests.get(
        image_path,
        timeout=10
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

else:

    image = Image.open(
        image_path
    ).convert("RGB")

device = torch.device("cpu")

with open(
    "label_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    label_mapping = json.load(file)

id_to_name = {
    class_id: mushroom_name
    for mushroom_name, class_id
    in label_mapping.items()
}

number_of_classes = len(label_mapping)

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=None
)


number_of_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.0),
    nn.Linear(
        number_of_features,
        number_of_classes
    )
)

model.load_state_dict(
    torch.load(
        local_model_path,
        map_location=device
    )
)


model.to(device)
model.eval()

transform = weights.transforms()

image_tensor = (
    transform(image)
    .unsqueeze(0)
    .to(device)
)

with torch.no_grad():
    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]

top_k = 3

top_probabilities, top_classes = torch.topk(
    probabilities,
    k=top_k
)

print()
print("=" * 58)
print("MUSHROOM CLASSIFICATION RESULTS")
print("=" * 58)
print(f"Image: {image_path}")
print("-" * 58)

for rank in range(top_k):

    class_id = top_classes[rank].item()
    probability = top_probabilities[rank].item() * 100
    mushroom_name = id_to_name[class_id]

    bar_length = int(probability / 2)
    bar = "█" * bar_length

    print(
        f"{rank + 1:>2}. "
        f"{mushroom_name:<28} "
        f"{probability:>6.2f}%  "
        f"{bar}"
    )

print("-" * 58)

best_class_id = top_classes[0].item()
best_probability = top_probabilities[0].item() * 100

print(
    f"Prediction: {id_to_name[best_class_id]} "
    f"({best_probability:.2f}%)"
)

print("=" * 58)