import pathlib
import yaml
import pandas as pd

from feature_definitions import feature_build


def load_data(data_path):
    return pd.read_csv(data_path)


def save_data(train, test, output_path):

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        output_path / "train.csv",
        index=False,
    )

    test.to_csv(
        output_path / "test.csv",
        index=False,
    )


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent

    with open(home_dir / "params.yaml") as f:
        params = yaml.safe_load(f)

    raw_dir = home_dir / params["data"]["raw_dir"]
    processed_dir = home_dir / params["data"]["processed_dir"]

    train_data = load_data(raw_dir / "train.csv")

    test_data = load_data(raw_dir / "test.csv")

    train_data = feature_build(
        train_data,
        "train",
        params,
    )

    test_data = feature_build(
        test_data,
        "test",
        params,
    )

    save_data(
        train_data,
        test_data,
        processed_dir,
    )


if __name__ == "__main__":
    main()
