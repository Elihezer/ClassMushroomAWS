# File logic
# → imports
# → Dataset
# → DataLoaders
# → ResNet18
# → train/validation loop
# → save best model
# → analyse misclassified validation images


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
#from torchvision.models import ResNet50_Weights, resnet50

# Alternative models for later tests.
# from torchvision.models import ResNet34_Weights, resnet34
# from torchvision.models import ResNet50_Weights, resnet50


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
        image_value = self.table[
            "image"
        ][index].as_py()

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
        bbox = self.table[
            "bbox"
        ][index].as_py()

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

        # Read the mushroom class.
        mushroom_name = self.table[
            "mushroom_name"
        ][index].as_py()

        # Convert the textual class into a numeric target.
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

    model_directory = Path(
        model_directory
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # ---------------------------------------------------------
    # Label mapping
    # ---------------------------------------------------------

    class_names = sorted(
        set(
            train_table[
                "mushroom_name"
            ].to_pylist()
        )
    )

    label_mapping = {
        mushroom_name: class_id
        for class_id, mushroom_name
        in enumerate(class_names)
    }

    number_of_classes = len(
        label_mapping
    )

    print(
        "Number of classes:",
        number_of_classes
    )

    print(
        "Label mapping:",
        label_mapping
    )


    # ---------------------------------------------------------
    # Pretrained ResNet18 preprocessing
    # ---------------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    image_transform = (
        weights.transforms()
    )


    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = MushroomDataset(
        train_table,
        label_mapping,
        image_transform
    )

    validation_dataset = MushroomDataset(
        validation_table,
        label_mapping,
        image_transform
    )


    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        )
    )


    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = resnet18(
        weights=weights
    )

    number_of_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Dropout(
            p=dropout_rate
        ),
        nn.Linear(
            number_of_features,
            number_of_classes
        )
    )


    # ---------------------------------------------------------
    # Freeze configuration
    # ---------------------------------------------------------

    # Start with every parameter trainable.
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


    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    print(
        "Training device:",
        device
    )


    # ---------------------------------------------------------
    # Loss functions
    # ---------------------------------------------------------

    training_loss_function = (
        nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )
    )

    validation_loss_function = (
        nn.CrossEntropyLoss()
    )


    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate
    )


    # ---------------------------------------------------------
    # Learning-rate scheduler
    # ---------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr
        )
    )


    # ---------------------------------------------------------
    # Training state
    # ---------------------------------------------------------

    training_history = []

    best_validation_accuracy = -1.0
    best_epoch = None

    model_path = (
        model_directory
        / "model.pt"
    )

    history_path = (
        model_directory
        / "training_history.json"
    )

    mapping_path = (
        model_directory
        / "label_mapping.json"
    )

    training_start_time = (
        time.perf_counter()
    )


    # =========================================================
    # TRAINING LOOP
    # =========================================================

    for epoch in range(epochs):

        epoch_start_time = (
            time.perf_counter()
        )

        # Enable training mode.
        model.train()

        # Keep frozen BatchNorm blocks in evaluation mode.
        if freeze_bn1:
            model.bn1.eval()

        if freeze_layer1:
            model.layer1.eval()

        if freeze_layer2:
            model.layer2.eval()


        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------

        training_loss = 0.0
        training_correct = 0
        training_total = 0

        for batch_index, (
            images,
            targets
        ) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = (
                training_loss_function(
                    outputs,
                    targets
                )
            )

            loss.backward()

            optimizer.step()

            training_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            training_correct += (
                predictions == targets
            ).sum().item()

            training_total += (
                targets.size(0)
            )

            if batch_index % 10 == 0:

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"| Batch "
                    f"{batch_index}/{len(train_loader)}"
                )


        epoch_training_loss = (
            training_loss
            / training_total
        )

        epoch_training_accuracy = (
            training_correct
            / training_total
        )


        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        model.eval()

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for (
                images,
                targets
            ) in validation_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                targets = targets.to(
                    device,
                    non_blocking=True
                )

                outputs = model(
                    images
                )

                loss = (
                    validation_loss_function(
                        outputs,
                        targets
                    )
                )

                validation_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = outputs.argmax(
                    dim=1
                )

                validation_correct += (
                    predictions == targets
                ).sum().item()

                validation_total += (
                    targets.size(0)
                )


        epoch_validation_loss = (
            validation_loss
            / validation_total
        )

        epoch_validation_accuracy = (
            validation_correct
            / validation_total
        )


        # -----------------------------------------------------
        # Scheduler
        # -----------------------------------------------------

        scheduler.step(
            epoch_validation_loss
        )

        print(
            "Learning rate for next epoch:",
            optimizer.param_groups[0]["lr"]
        )


        # -----------------------------------------------------
        # Epoch timing
        # -----------------------------------------------------

        epoch_duration_seconds = (
            time.perf_counter()
            - epoch_start_time
        )

        print(
            f"Epoch duration: "
            f"{epoch_duration_seconds:.2f} seconds"
        )


        # -----------------------------------------------------
        # Store epoch metrics
        # -----------------------------------------------------

        epoch_statistics = {
            "epoch": epoch + 1,
            "train_loss":
                epoch_training_loss,
            "train_accuracy":
                epoch_training_accuracy,
            "validation_loss":
                epoch_validation_loss,
            "validation_accuracy":
                epoch_validation_accuracy,
            "epoch_duration_seconds":
                epoch_duration_seconds
        }

        training_history.append(
            epoch_statistics
        )


        # -----------------------------------------------------
        # Save training history
        # -----------------------------------------------------

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


        # -----------------------------------------------------
        # Display epoch results
        # -----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train loss: "
            f"{epoch_training_loss:.4f} "
            f"| Train accuracy: "
            f"{epoch_training_accuracy:.4f} "
            f"| Validation loss: "
            f"{epoch_validation_loss:.4f} "
            f"| Validation accuracy: "
            f"{epoch_validation_accuracy:.4f}"
        )


        # -----------------------------------------------------
        # Save best model
        # -----------------------------------------------------

        if (
            epoch_validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                epoch_validation_accuracy
            )

            best_epoch = (
                epoch + 1
            )

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


    # =========================================================
    # END OF TRAINING
    # =========================================================


    # ---------------------------------------------------------
    # Save label mapping
    # ---------------------------------------------------------

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as mapping_file:

        json.dump(
            label_mapping,
            mapping_file,
            indent=4,
            ensure_ascii=False
        )


    training_duration_seconds = (
        time.perf_counter()
        - training_start_time
    )

    print(
        "Training completed."
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation accuracy:",
        best_validation_accuracy
    )


    # =========================================================
    # MISCLASSIFIED VALIDATION IMAGES
    # =========================================================


    # Reload the best model.
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()


    # Reverse label mapping:
    # numeric ID → mushroom name.
    inverse_label_mapping = {
        class_id: mushroom_name
        for mushroom_name, class_id
        in label_mapping.items()
    }


    # Directory containing validation failures.
    misclassified_directory = (
        model_directory
        / "misclassified"
    )

    misclassified_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    misclassified_count = 0


    with torch.no_grad():

        for index in range(
            validation_table.num_rows
        ):

            # Read encoded image.
            image_value = (
                validation_table[
                    "image"
                ][index].as_py()
            )

            if isinstance(
                image_value,
                dict
            ):
                image_bytes = (
                    image_value["bytes"]
                )
            else:
                image_bytes = (
                    image_value
                )


            # Decode original image.
            image = Image.open(
                BytesIO(
                    image_bytes
                )
            ).convert(
                "RGB"
            )


            # Read bounding box.
            bbox = validation_table[
                "bbox"
            ][index].as_py()

            (
                x_min,
                y_min,
                x_max,
                y_max
            ) = map(
                int,
                bbox
            )


            # Crop the mushroom.
            cropped_image = image.crop(
                (
                    x_min,
                    y_min,
                    x_max,
                    y_max
                )
            )


            # Apply model preprocessing.
            model_input = (
                image_transform(
                    cropped_image
                )
                .unsqueeze(0)
                .to(device)
            )


            # Inference.
            outputs = model(
                model_input
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predicted_class_id = (
                outputs.argmax(
                    dim=1
                ).item()
            )


            confidence = (
                probabilities[
                    0,
                    predicted_class_id
                ].item()
            )


            # Read true label.
            true_label = (
                validation_table[
                    "mushroom_name"
                ][index].as_py()
            )


            true_class_id = (
                label_mapping[
                    true_label
                ]
            )


            # Save only errors.
            if (
                predicted_class_id
                != true_class_id
            ):

                predicted_label = (
                    inverse_label_mapping[
                        predicted_class_id
                    ]
                )


                true_label_safe = (
                    true_label.replace(
                        " ",
                        "_"
                    )
                )

                predicted_label_safe = (
                    predicted_label.replace(
                        " ",
                        "_"
                    )
                )


                filename = (
                    f"{index:04d}"
                    f"_true_{true_label_safe}"
                    f"_pred_{predicted_label_safe}"
                    f"_conf_{confidence:.2f}.jpg"
                )


                cropped_image.save(
                    misclassified_directory
                    / filename
                )


                misclassified_count += 1


    print(
        "Misclassified validation images:",
        misclassified_count
    )

    print(
        "Misclassified images saved to:",
        misclassified_directory
    )


    # =========================================================
    # RETURN TRAINING RESULTS
    # =========================================================

    return {
        "model_path":
            model_path,

        "mapping_path":
            mapping_path,

        "history_path":
            history_path,

        "best_validation_accuracy":
            best_validation_accuracy,

        "best_epoch":
            best_epoch,

        "training_history":
            training_history,

        "training_duration_seconds":
            training_duration_seconds,

        "number_of_classes":
            number_of_classes,

        "optimizer":
            optimizer.__class__.__name__,

        "training_loss_function":
            training_loss_function.__class__.__name__,

        "validation_loss_function":
            validation_loss_function.__class__.__name__,

        "pretrained_weights":
            str(weights),

        "device":
            str(device),

        "misclassified_directory":
            str(
                misclassified_directory
            )
    }