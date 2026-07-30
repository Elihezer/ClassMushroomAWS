# ClassMushroomAWS

AWS execution code for the ClassMushroom machine-learning project.

## Architecture

- Training data: Amazon S3
- Remote training: Amazon SageMaker AI
- Logs: Amazon CloudWatch
- Model artifacts: Amazon S3

## Source code

- `src/train.py`: code executed by SageMaker
- `launcher/launch_training.py`: submits the SageMaker training job