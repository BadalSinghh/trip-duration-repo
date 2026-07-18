import pathlib
import yaml
import pandas as pd
import numpy as np


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    avg_earth_radius = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    return 2 * avg_earth_radius * np.arcsin(np.sqrt(d))


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(
        np.radians,
        (lat1, lng1, lat2, lng2),
    )

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))


def datetime_feature_fix(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_date"] = df["pickup_datetime"].dt.date
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)


def create_dist_features(df):
    df["distance_haversine"] = haversine_array(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["distance_dummy_manhattan"] = dummy_manhattan_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["direction"] = bearing_array(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )


def create_datetime_features(df):
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_minute"] = df["pickup_datetime"].dt.minute
    df["pickup_dt"] = (df["pickup_datetime"] - df["pickup_datetime"].min()).dt.total_seconds()
    df["pickup_week_hour"] = df["pickup_weekday"] * 24 + df["pickup_hour"]


def feature_build(df, tag, params):

    datetime_feature_fix(df)
    create_dist_features(df)
    create_datetime_features(df)

    drop_columns = params["features"]["drop_columns"]

    feature_names = [col for col in df.columns if col not in drop_columns]

    print(f"We have {len(feature_names)} features in {tag}.")

    return df[feature_names]


if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent

    with open(home_dir / "params.yaml") as f:
        params = yaml.safe_load(f)

    data = pd.read_csv(
        home_dir / "data" / "raw" / "test.csv",
        nrows=10,
    )

    data = feature_build(
        data,
        "test",
        params,
    )

    print(data.head())
