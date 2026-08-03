#File logic
#→ imports
#→ arguments
#→ SageMaker paths
#→ Parquet loading
#→ label mapping
#→ Dataset
#→ DataLoaders
#→ ResNet18
#→ train/validation loop
#→ save model

# Standard library imports.
import argparse
import json
import os
from io import BytesIO
from pathlib import Path

# Third-party imports.
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


# Read hyperparameters passed by the SageMaker Training Job.
parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs",
    type=int,
    default=10
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=32
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.001
)

args = parser.parse_args()


# Read the directories automatically created by SageMaker.
train_directory = Path(
    os.environ.get(
        "SM_CHANNEL_TRAIN",
        "/opt/ml/input/data/train"
    )
)

validation_directory = Path(
    os.environ.get(
        "SM_CHANNEL_VALIDATION",
        "/opt/ml/input/data/validation"
    )
)

model_directory = Path(
    os.environ.get(
        "SM_MODEL_DIR",
        "/opt/ml/model"
    )
)

# Create the directory used to store the final model artifacts.
model_directory.mkdir(
    parents=True,
    exist_ok=True
)

# Find the Parquet file downloaded in the training channel.
train_files = list(
    train_directory.glob("*.parquet")
)

# Find the Parquet file downloaded in the validation channel.
validation_files = list(
    validation_directory.glob("*.parquet")
)


# Stop the training job if the training file is missing.
if len(train_files) != 1:
    raise RuntimeError(
        f"Expected exactly one training Parquet file, "
        f"but found {len(train_files)} in {train_directory}"
    )

# Stop the training job if the validation file is missing.
if len(validation_files) != 1:
    raise RuntimeError(
        f"Expected exactly one validation Parquet file, "
        f"but found {len(validation_files)} in "
        f"{validation_directory}"
    )


# Select the detected Parquet files.
train_path = train_files[0]
validation_path = validation_files[0]


# Display the resolved SageMaker paths.
print("Training file:", train_path)
print("Validation file:", validation_path)
print("Model output directory:", model_directory)

# Display the received hyperparameters.
print("Epochs:", args.epochs)
print("Batch size:", args.batch_size)
print("Learning rate:", args.learning_rate)


# Read the training and validation Parquet datasets.
train_table = pq.read_table(train_path)
validation_table = pq.read_table(validation_path)


# Display basic dataset information.
print("Training rows:", train_table.num_rows)
print("Validation rows:", validation_table.num_rows)

print("Training columns:", train_table.column_names)
print("Validation columns:", validation_table.column_names)

# Extract the unique mushroom class names from the training dataset.
class_names = sorted(
    set(
        train_table["mushroom_name"].to_pylist()
    )
)

# Assign one numeric ID to each mushroom class.
label_mapping = {
    mushroom_name: class_id
    for class_id, mushroom_name in enumerate(class_names)
}

number_of_classes = len(label_mapping)

print("Number of classes:", number_of_classes)
print("Label mapping:", label_mapping)

class MushroomDataset(Dataset):
    def __init__(
        self,
        table,
        label_mapping,
        transform
    ):
        self.table = table
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, index):
        # Read the encoded image structure from the Arrow table.
        image_value = self.table["image"][index].as_py()

        # Extract the encoded image bytes.
        if isinstance(image_value, dict):
            image_bytes = image_value["bytes"]
        else:
            image_bytes = image_value

        # Decode the image and convert it to RGB.
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # Read the bounding box coordinates.
        bbox = self.table["bbox"][index].as_py()

        x_min, y_min, x_max, y_max = map(
            int,
            bbox
        )

        # Crop the image around the mushroom.
        image = image.crop(
            (
                x_min,
                y_min,
                x_max,
                y_max
            )
        )

        # Apply the preprocessing required by ResNet18.
        image = self.transform(image)

        # Convert the textual mushroom name into a numeric target.
        mushroom_name = self.table[
            "mushroom_name"
        ][index].as_py()

        target = self.label_mapping[
            mushroom_name
        ]

        return image, target


# Load the default pretrained ImageNet weights.
weights = ResNet18_Weights.DEFAULT

# Load the preprocessing pipeline associated with these weights.
image_transform = weights.transforms()

# Create the training dataset.
train_dataset = MushroomDataset(
    train_table,
    label_mapping,
    image_transform
)

# Create the validation dataset.
validation_dataset = MushroomDataset(
    validation_table,
    label_mapping,
    image_transform
)

# Create the training DataLoader.
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0
)


# Create the validation DataLoader.
validation_loader = DataLoader(
    validation_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0
)

# Load ResNet18 with pretrained ImageNet weights.
model = resnet18(
    weights=weights
)

# Replace the original ImageNet classification layer.
model.fc = nn.Linear(
    model.fc.in_features,
    number_of_classes
)

# Use a GPU when available, otherwise use the CPU.
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

print("Training device:", device)

# Define the loss function for multi-class classification.
loss_function = nn.CrossEntropyLoss()

# Define the optimizer.
optimizer = AdamW(
    model.parameters(),
    lr=args.learning_rate
)

# Start below every possible accuracy value so the first model is saved.
best_validation_accuracy = -1.0

for epoch in range(args.epochs):
    # Enable training mode.
    model.train()

    training_loss = 0.0
    training_correct = 0
    training_total = 0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(
            outputs,
            targets
        )

        loss.backward()
        optimizer.step()

        training_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        training_correct += (
            predictions == targets
        ).sum().item()

        training_total += targets.size(0)

    epoch_training_loss = (
        training_loss / training_total
    )

    epoch_training_accuracy = (
        training_correct / training_total
    )

    # Enable evaluation mode.
    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():
        for images, targets in validation_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss = loss_function(
                outputs,
                targets
            )

            validation_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            validation_correct += (
                predictions == targets
            ).sum().item()

            validation_total += targets.size(0)

    epoch_validation_loss = (
        validation_loss / validation_total
    )

    epoch_validation_accuracy = (
        validation_correct / validation_total
    )

    print(
        f"Epoch {epoch + 1}/{args.epochs} "
        f"| Train loss: {epoch_training_loss:.4f} "
        f"| Train accuracy: {epoch_training_accuracy:.4f} "
        f"| Validation loss: {epoch_validation_loss:.4f} "
        f"| Validation accuracy: {epoch_validation_accuracy:.4f}"
    )

    if epoch_validation_accuracy > best_validation_accuracy:
        # Update the best validation accuracy.
        best_validation_accuracy = epoch_validation_accuracy
    
        # Define the output path of the best model.
        model_path = model_directory / "model.pt"
    
        # Save the weights of the best model.
        torch.save(
            model.state_dict(),
            model_path
        )
    
        print(
            "Best model saved:",
            model_path,
            "| Validation accuracy:",
            best_validation_accuracy
        )
    
# Save the label mapping next to the trained model.
with open(
    model_directory / "label_mapping.json",
    "w",
    encoding="utf-8"
) as mapping_file:
    json.dump(
        label_mapping,
        mapping_file,
        indent=4,
        ensure_ascii=False
    )

print("Training completed.")
print("Best validation accuracy:", best_validation_accuracy)