#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Use both available vCPUs for PyTorch CPU operations.
torch.set_num_threads(64)

# Keep inter-operation parallelism simple.
torch.set_num_interop_threads(1)

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


# In[2]:


# Define the first training configuration.
epochs = 20
batch_size = 32
learning_rate = 0.0001   # dynamic
num_workers = 8
label_smoothing = 0.0

print("Epochs:", epochs)
print("Batch size:", batch_size)
print("Learning rate:", learning_rate)
print("DataLoader workers:", num_workers)


# In[3]:


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


# In[4]:


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


# In[5]:


# Extract the unique class names from the training dataset.
class_names = sorted(
    set(
        train_table["mushroom_name"].to_pylist()
    )
)

# Assign one numeric ID to every mushroom class.
label_mapping = {
    mushroom_name: class_id
    for class_id, mushroom_name
    in enumerate(class_names)
}

number_of_classes = len(label_mapping)

print("Number of classes:", number_of_classes)
print("Label mapping:")
label_mapping


# In[6]:


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
        # Read the encoded image value.
        image_value = self.table[
            "image"
        ][index].as_py()

        # Extract the encoded image bytes.
        if isinstance(image_value, dict):
            image_bytes = image_value["bytes"]
        else:
            image_bytes = image_value

        # Decode the image as RGB.
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # Read the bounding box.
        bbox = self.table[
            "bbox"
        ][index].as_py()

        x_min, y_min, x_max, y_max = map(
            int,
            bbox
        )

        # Crop the image around the annotated mushroom.
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

        # Read the textual target.
        mushroom_name = self.table[
            "mushroom_name"
        ][index].as_py()

        # Convert the textual target into a numeric ID.
        target = self.label_mapping[
            mushroom_name
        ]

        return image, target


# In[7]:


# Select the default pretrained ResNet18 weights.
weights = ResNet18_Weights.DEFAULT

# Load the preprocessing associated with these weights.
image_transform = weights.transforms()

print("ResNet18 preprocessing:")
image_transform


# In[8]:


# Create the training Dataset.
train_dataset = MushroomDataset(
    train_table,
    label_mapping,
    image_transform
)

# Create the validation Dataset.
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

print("Training batches:", len(train_loader))
print("Validation batches:", len(validation_loader))

print("PyTorch threads:", torch.get_num_threads())
print("Inter-op threads:", torch.get_num_interop_threads())
print("DataLoader workers:", train_loader.num_workers)


# In[9]:


# Read one training batch.
sample_images, sample_targets = next(
    iter(train_loader)
)

print("Image batch shape:", sample_images.shape)
print("Target batch shape:", sample_targets.shape)

print("Image tensor type:", sample_images.dtype)
print("Target tensor type:", sample_targets.dtype)

print("First targets:", sample_targets[:10])


# In[10]:


# Load ResNet18 with pretrained ImageNet weights.
model = resnet18(
    weights=weights
)

# Define the dropout probability.
dropout_rate = 0.0

# Save the number of features produced by ResNet18.
number_of_features = model.fc.in_features

# Replace the original classifier with dropout and a new classifier.
model.fc = nn.Sequential(
    nn.Dropout(p=dropout_rate),
    nn.Linear(
        number_of_features,
        number_of_classes
    )
)

# Ensure that every parameter is initially trainable.                                 
# This is useful when the notebook cells are executed several times.
for parameter in model.parameters():
    parameter.requires_grad = True

# Freeze the first convolution layer.
for parameter in model.conv1.parameters():
    parameter.requires_grad = False

# Freeze the first batch-normalization layer.
for parameter in model.bn1.parameters():
    parameter.requires_grad = False

# Freeze the first ResNet block.
for parameter in model.layer1.parameters():
    parameter.requires_grad = False


# Freeze the second residual stage.
for parameter in model.layer2.parameters():
    parameter.requires_grad = False    

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")
print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.2f}%"
)

# Move the model to the selected device.
model = model.to(device)

print(model.fc)
print("Model device:", device)


# In[11]:


# Define the regularized loss used during training.
training_loss_function = nn.CrossEntropyLoss(
    label_smoothing=label_smoothing
)

# Define the standard loss used during validation.
validation_loss_function = nn.CrossEntropyLoss()

# Define the optimizer.
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)

# Define the learning-rate scheduler.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",       # Reduce the LR when validation loss stops decreasing.
    factor=0.5,       # Divide the LR by two.
    patience=1,       # Allow one epoch without improvement.
    min_lr=0.000001   # Do not reduce the LR below 1e-6.
)

print(    "Training loss function:",    training_loss_function.__class__.__name__)
print(    "Training label smoothing:",    label_smoothing)
print(    "Validation loss function:",    validation_loss_function.__class__.__name__)
print(    "Optimizer:",    optimizer.__class__.__name__)
print(    "Initial learning rate:",    optimizer.param_groups[0]["lr"])


# In[12]:


# Standard library import.
from datetime import datetime


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

# Define the artifact paths for this training run.
model_path = (
    run_directory
    / "model.pt"
)

mapping_path = (
    run_directory
    / "label_mapping.json"
)

history_path = (
    run_directory
    / "training_history.json"
)

print("Training run ID:", run_id)
print("Run directory:", run_directory)
print("Model path:", model_path)
print("Mapping path:", mapping_path)
print("History path:", history_path)


# In[13]:


# Standard library imports.
import json
import platform
import time

# Store the configuration used by this training run.
run_config = {
    "run_id": run_id,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_workers": num_workers,
    "optimizer": optimizer.__class__.__name__,
    "training_loss_function": training_loss_function.__class__.__name__,
    "training_label_smoothing": label_smoothing,
    "validation_loss_function": validation_loss_function.__class__.__name__,
    "model": "ResNet18",
    "pretrained_weights": str(weights),
    "number_of_classes": number_of_classes,
    "training_rows": train_table.num_rows,
    "validation_rows": validation_table.num_rows,
    "device": str(device),
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

# Initialize the epoch history.
training_history = []



print("Run configuration saved to:", config_path)


# In[14]:


# Record the run start time.
training_start_time = time.perf_counter()

# Start below every possible accuracy value.
best_validation_accuracy = -1.0

# Keep track of the epoch that produced the best model.
best_epoch = None

for epoch in range(epochs):

    epoch_start_time = time.perf_counter()
    # Enable training mode.
    model.train()

    # Keep the frozen BatchNorm layers in evaluation mode.
    model.bn1.eval()
    model.layer1.eval()

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

        training_total += targets.size(0)

        # Display progress every ten batches.
        if batch_index % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Batch {batch_index}/{len(train_loader)}"
            )

    epoch_training_loss = (
        training_loss
        / training_total
    )

    epoch_training_accuracy = (
        training_correct
        / training_total
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
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            validation_correct += (
                predictions == targets
            ).sum().item()

            validation_total += targets.size(0)

    epoch_validation_loss = (
        validation_loss
        / validation_total
    )

    # Give the validation loss to the scheduler.
    scheduler.step(epoch_validation_loss)
    
 

    # Calculate the complete epoch duration.
    epoch_duration_seconds = (    time.perf_counter()    - epoch_start_time    )

    # Display the learning rate that will be used for the next epoch.
    print( f"Time to complete Epoch: {epoch_duration_seconds} seconds")

    # Display the learning rate that will be used for the next epoch.
    print(
        "Learning rate for next epoch",
        optimizer.param_groups[0]["lr"]
    )

    epoch_validation_accuracy = (
        validation_correct
        / validation_total
    )

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| Train loss: {epoch_training_loss:.4f} "
        f"| Train accuracy: {epoch_training_accuracy:.4f} "
        f"| Validation loss: {epoch_validation_loss:.4f} "
        f"| Validation accuracy: "
        f"{epoch_validation_accuracy:.4f}"
    )
    # Store the metrics produced by the current epoch.
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
    
    # Save the complete history after every epoch.
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
        "Training history updated:",
        history_path
    )

    # Save the best model weights.
    if (
        epoch_validation_accuracy
        > best_validation_accuracy
    ):
        best_validation_accuracy = (
            epoch_validation_accuracy
        )
    
        best_epoch = epoch + 1
    
        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            "Best model saved:",
            model_path
        )


# In[15]:


# Save the label mapping for this training run.
# Measure the complete training duration.
training_duration_seconds = (
    time.perf_counter()
    - training_start_time
)

# Save the label mapping for this run.
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


# In[16]:


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


# In[ ]:




