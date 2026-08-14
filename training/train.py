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

import json
import time

from io import BytesIO
from pathlib import Path

# Third-party imports.
import torch
import torch.nn as nn

from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


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

def train(
    train_table,
    validation_table,
    model_directory,
    epochs,
    batch_size,
    learning_rate,
    num_workers,
    label_smoothing,
    dropout_rate,
    freeze_conv1,
    freeze_bn1,
    freeze_layer1,
    freeze_layer2,
    scheduler_factor,
    scheduler_patience,
    scheduler_min_lr
):
    model_directory = Path(model_directory)

    # Extract the unique mushroom class names from the training dataset.
    class_names = sorted(
        set(
            train_table["mushroom_name"].to_pylist()
        )
    )

    label_mapping = {
        mushroom_name: class_id
        for class_id, mushroom_name in enumerate(class_names)
    }

    number_of_classes = len(label_mapping)

    print("Number of classes:", number_of_classes)
    print("Label mapping:", label_mapping)

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


    # Create the validation DataLoader.
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Load ResNet18 with pretrained ImageNet weights.
    model = resnet18(
        weights=weights
    )

    # Replace the original ImageNet classification layer.
    number_of_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(
            number_of_features,
            number_of_classes
        )
    )

    # Ensure that all model parameters are initially trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    if freeze_conv1:
        for parameter in model.conv1.parameters():
            parameter.requires_grad = False

    if freeze_bn1:
        for parameter in model.bn1.parameters():
            parameter.requires_grad = False

    if freeze_layer1:
        for parameter in model.layer1.parameters():
            parameter.requires_grad = False

    if freeze_layer2:
        for parameter in model.layer2.parameters():
            parameter.requires_grad = False

    # Use a GPU when available, otherwise use the CPU.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    print("Training device:", device)

    # Define the loss function for multi-class classification.
    training_loss_function = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    validation_loss_function = nn.CrossEntropyLoss()

    # Define the optimizer.
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate
    )

    # Define the scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )

    training_history = []
    best_epoch = None

    training_start_time = time.perf_counter()

    # Start below every possible accuracy value so the first model is saved.
    best_validation_accuracy = -1.0

    for epoch in range(epochs):
        # Enable training mode.
        model.train()

        if freeze_bn1:
            model.bn1.eval()

        if freeze_layer1:
            model.layer1.eval()

        if freeze_layer2:
            model.layer2.eval() 
        

        training_loss = 0.0
        training_correct = 0
        training_total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = training_loss_function(
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

                loss = validation_loss_function(
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

        scheduler.step(epoch_validation_loss)

        epoch_validation_accuracy = (
            validation_correct / validation_total
        )

        epoch_statistics = {
            "epoch": epoch + 1,
            "train_loss": epoch_training_loss,
            "train_accuracy": epoch_training_accuracy,
            "validation_loss": epoch_validation_loss,
            "validation_accuracy": epoch_validation_accuracy
        }

        training_history.append(
            epoch_statistics
        )

        history_path = (
            model_directory
            / "training_history.json"
        )

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as history_file:  
            json.dump(
                training_history,
                history_file,
                indent=4
            )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train loss: {epoch_training_loss:.4f} "
            f"| Train accuracy: {epoch_training_accuracy:.4f} "
            f"| Validation loss: {epoch_validation_loss:.4f} "
            f"| Validation accuracy: {epoch_validation_accuracy:.4f}"
        )

        if epoch_validation_accuracy > best_validation_accuracy:
            # Update the best validation accuracy.
            best_validation_accuracy = epoch_validation_accuracy

            best_epoch = epoch + 1
        
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

    training_duration_seconds = (
        time.perf_counter() - training_start_time
    )

    return {
        "model_path": model_directory / "model.pt",
        "mapping_path": model_directory / "label_mapping.json",
        "history_path": model_directory / "training_history.json",

        "best_validation_accuracy": best_validation_accuracy,
        "best_epoch": best_epoch,

        "training_history": training_history,
        "training_duration_seconds": training_duration_seconds,

        "number_of_classes": number_of_classes,

        "optimizer": optimizer.__class__.__name__,
        "training_loss_function": (
            training_loss_function.__class__.__name__
        ),
        "validation_loss_function": (
            validation_loss_function.__class__.__name__
        ),

        "pretrained_weights": str(weights),
        "device": str(device)
    }