import pathlib
import pandas as pd

from feature_definitions import feature_build


def load_data(data_path):
    """Load dataset from the given path."""
    return pd.read_csv(data_path)


def save_data(train, test, output_path):
    """Save processed train and test datasets."""
    output_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)


if __name__ == "__main__":
    # Current file: project_root/trip_duration/build_features.py
    curr_dir = pathlib.Path(__file__)

    # Move up to the project root
    home_dir = curr_dir.parent.parent

    # Define paths
    train_path = home_dir / "data" / "raw" / "train.csv"
    test_path = home_dir / "data" / "raw" / "test.csv"
    output_path = home_dir / "data" / "processed"

    # Load data
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    # Build features
    train_data = feature_build(train_data, "train")
    test_data = feature_build(test_data, "test")

    # Save processed data
    save_data(train_data, test_data, output_path)
