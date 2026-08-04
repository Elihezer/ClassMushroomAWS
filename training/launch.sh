#!/bin/bash
set -e

sudo dnf install -y python3-pip
python3 -m pip install -r requirements.txt jupyter
python3 -m jupyter nbconvert --to notebook --execute training/01B_launch_training_inNotebook.ipynb --output training_completed.ipynb --ExecutePreprocessor.timeout=-1
