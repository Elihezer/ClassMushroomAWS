import sys
import json

import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


import boto3


bucket_name = "mushroom-ml-mikael-2026-000"
model_key = "inference/model/v1/model.pt"
local_model_path = "model.pt"

s3_client = boto3.client("s3")

s3_client.download_file(
    bucket_name,
    model_key,
    local_model_path
)

model.load_state_dict(
    torch.load(
        local_model_path,
        map_location=device
    )
)


image_path = sys.argv[1]

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
        "model.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

transform = weights.transforms()

image = Image.open(
    image_path
).convert("RGB")

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

top_probabilities, top_classes = torch.topk(
    probabilities,
    k=3
)

for rank in range(3):

    class_id = top_classes[rank].item()

    probability = (
        top_probabilities[rank].item()
        * 100
    )

    print(
        f"{rank + 1}. "
        f"{id_to_name[class_id]} "
        f"- {probability:.2f}%"
    )