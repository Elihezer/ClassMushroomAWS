#!/usr/bin/env python
# coding: utf-8

# Import of the train function from the train.py file.
from train import train

# Standard library imports.
import json
from io import BytesIO
from pathlib import Path

# Third-party imports.
import boto3
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from PIL import Image
from pyarrow import fs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

# Standard library import.
from datetime import datetime


# Standard library imports.
import json
import platform
import time



# Read the current notebook directory.
current_directory = Path.cwd()

# Detect the AWS Region used by the notebook.
aws_region = boto3.Session().region_name

# Create the PyArrow filesystem used to access Amazon S3.
s3 = fs.S3FileSystem(
    region=aws_region
)

# Select the training device.
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Current directory:", current_directory)
print("AWS Region:", aws_region)
print("PyTorch version:", torch.__version__)
print("Training device:", device)

# -------------------------------------------------------------------------
# Runtime configuration
# -------------------------------------------------------------------------

torch_threads = 4
interop_threads = 1
num_workers = 2

# -------------------------------------------------------------------------
# Training configuration
# -------------------------------------------------------------------------

epochs = 12
batch_size = 32
learning_rate = 0.0001

label_smoothing = 0.0
dropout_rate = 0.0

# -------------------------------------------------------------------------
# Transfer-learning configuration
# -------------------------------------------------------------------------

freeze_conv1 = True
freeze_bn1 = True
freeze_layer1 = True
freeze_layer2 = True

# -------------------------------------------------------------------------
# Learning-rate scheduler configuration
# -------------------------------------------------------------------------

scheduler_factor = 0.5
scheduler_patience = 1
scheduler_min_lr = 0.000001

# Setting concerning how many threads to use for PyTorch CPU operations.
torch.set_num_threads(torch_threads)

# Keep inter-operation parallelism simple.
torch.set_num_interop_threads(interop_threads)


# Define the S3 bucket name.
bucket_name = "mushroom-ml-mikael-2026-000"

# Define the training Parquet path.
train_path = (
    f"{bucket_name}/processed/parquet/"
    "train_balanced_augmented.parquet"
)

# Define the validation Parquet path.
validation_path = (
    f"{bucket_name}/processed/parquet/"
    "validation.parquet"
)

print("Training path:", train_path)
print("Validation path:", validation_path)


# Read the training dataset from Amazon S3.
train_table = pq.read_table(
    train_path,
    filesystem=s3
)

# Read the validation dataset from Amazon S3.
validation_table = pq.read_table(
    validation_path,
    filesystem=s3
)

print("Training rows:", train_table.num_rows)
print("Validation rows:", validation_table.num_rows)

print("Training columns:", train_table.column_names)
print("Validation columns:", validation_table.column_names)

# Create a unique identifier for this training run.
run_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

# Create one dedicated directory for this training run.
run_directory = (
    current_directory
    / "model_output"
    / run_id
)

run_directory.mkdir(
    parents=True,
    exist_ok=False
)

training_result = train(
    train_table=train_table,
    validation_table=validation_table,
    model_directory=run_directory,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    num_workers=num_workers,
    label_smoothing=label_smoothing,
    dropout_rate=dropout_rate,
    freeze_conv1=freeze_conv1,
    freeze_bn1=freeze_bn1,
    freeze_layer1=freeze_layer1,
    freeze_layer2=freeze_layer2,
    scheduler_factor=scheduler_factor,
    scheduler_patience=scheduler_patience,
    scheduler_min_lr=scheduler_min_lr
)

# Get results produced by train().
model_path = training_result["model_path"]
mapping_path = training_result["mapping_path"]
history_path = training_result["history_path"]

best_validation_accuracy = (
    training_result["best_validation_accuracy"]
)

best_epoch = training_result["best_epoch"]

training_history = (
    training_result["training_history"]
)

training_duration_seconds = (
    training_result["training_duration_seconds"]
)

number_of_classes = (
    training_result["number_of_classes"]
)

print("Training run ID:", run_id)
print("Run directory:", run_directory)
print("Model path:", model_path)
print("Mapping path:", mapping_path)
print("History path:", history_path)




# Store the configuration used by this training run.
run_config = {
    "run_id": run_id,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_workers": num_workers,
    "optimizer": training_result["optimizer"],
    "training_loss_function": training_result["training_loss_function"],
    "training_label_smoothing": label_smoothing,
    "model": "ResNet18",
    "pretrained_weights": training_result["pretrained_weights"],
    "number_of_classes": number_of_classes,
    "training_rows": train_table.num_rows,
    "validation_rows": validation_table.num_rows,
    "device": training_result["device"],
    "python_version": platform.python_version(),
    "pytorch_version": torch.__version__,
    "dropout_rate": dropout_rate
}

# Define the statistics output paths.
config_path = run_directory / "run_config.json"
history_path = run_directory / "training_history.json"
summary_path = run_directory / "run_summary.json"

# Save the run configuration before training starts.
with open(
    config_path,
    "w",
    encoding="utf-8"
) as config_file:
    json.dump(
        run_config,
        config_file,
        indent=4,
        ensure_ascii=False
    )


print("Run configuration saved to:", config_path)


# Build the final run summary.
run_summary = {
    "run_id": run_id,
    "completed_epochs": len(training_history),
    "best_epoch": best_epoch,
    "best_validation_accuracy": best_validation_accuracy,
    "final_train_loss": training_history[-1]["train_loss"],
    "final_train_accuracy": training_history[-1]["train_accuracy"],
    "final_validation_loss": training_history[-1][
        "validation_loss"
    ],
    "final_validation_accuracy": training_history[-1][
        "validation_accuracy"
    ],
    "training_duration_seconds": training_duration_seconds,
    "training_duration_minutes": (
        training_duration_seconds / 60
    ),
    "model_path": str(model_path),
    "mapping_path": str(mapping_path)
}

# Save the final run summary.
with open(
    summary_path,
    "w",
    encoding="utf-8"
) as summary_file:
    json.dump(
        run_summary,
        summary_file,
        indent=4,
        ensure_ascii=False
    )

print("Training completed.")
print("Best epoch:", best_epoch)
print(
    "Best validation accuracy:",
    best_validation_accuracy
)
print(
    "Training duration:",
    f"{training_duration_seconds / 60:.2f} minutes"
)

print("Run directory:", run_directory)
print("Model saved to:", model_path)
print("Mapping saved to:", mapping_path)
print("Configuration saved to:", config_path)
print("History saved to:", history_path)
print("Summary saved to:", summary_path)




# Define the S3 destination for this training run.
s3_run_prefix = (
    f"training/notebook_runs/{run_id}"
)

# Create the S3 client.
s3_client = boto3.client("s3")

# Define every artifact produced by the training run.
artifacts_to_upload = {
    "model.pt": model_path,
    "label_mapping.json": mapping_path,
    "run_config.json": config_path,
    "training_history.json": history_path,
    "run_summary.json": summary_path
}

# Upload every run artifact to Amazon S3.
for artifact_name, local_path in artifacts_to_upload.items():
    s3_client.upload_file(
        str(local_path),
        bucket_name,
        f"{s3_run_prefix}/{artifact_name}"
    )

    print(
        "Uploaded:",
        artifact_name
    )

print(
    "Training run uploaded to:",
    f"s3://{bucket_name}/{s3_run_prefix}/"
)

