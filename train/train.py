import os
from pathlib import Path


def main() -> None:
    train_directory = Path(
        os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    )

    model_directory = Path(
        os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )

    print(f"Training data directory: {train_directory}")
    print(f"Model output directory: {model_directory}")

    if train_directory.exists():
        print("Training files:")

        for file_path in train_directory.rglob("*"):
            if file_path.is_file():
                print(file_path)
    else:
        print("Training directory does not exist.")

    model_directory.mkdir(parents=True, exist_ok=True)

    test_file = model_directory / "aws_execution_test.txt"
    test_file.write_text(
        "The training script was successfully executed by SageMaker.",
        encoding="utf-8",
    )

    print(f"Created model artifact: {test_file}")


if __name__ == "__main__":
    main()