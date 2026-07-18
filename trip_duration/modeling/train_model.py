import pathlib
import sys
import joblib
import mlflow
import pandas as pd

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import mlflow.xgboost


TARGET = "trip_duration"


def find_best_model_with_params(X_train, y_train, X_test, y_test):

    space = {
        "n_estimators": hp.choice("n_estimators", [10, 15, 20]),
        "max_depth": hp.choice("max_depth", [6, 8, 10]),
        "learning_rate": hp.uniform("learning_rate", 0.03, 0.30),
    }

    def evaluate_model(params):

        params["max_depth"] = int(params["max_depth"])

        # Every Hyperopt trial gets its own MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = XGBRegressor(
                random_state=42,
                **params,
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            rmse = root_mean_squared_error(
                y_test,
                predictions,
            )

            mlflow.log_metric("RMSE", rmse)

            return {
                "loss": rmse,
                "status": STATUS_OK,
            }

    # Parent run
    with mlflow.start_run(run_name="XGBRegressor"):
        trials = Trials()

        best = fmin(
            fn=evaluate_model,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
        )

        best_params = space_eval(space, best)

        best_params["max_depth"] = int(best_params["max_depth"])

        mlflow.log_params(best_params)

        model = XGBRegressor(
            random_state=42,
            **best_params,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = root_mean_squared_error(
            y_test,
            predictions,
        )

        mlflow.log_metric("Final RMSE", rmse)

        mlflow.xgboost.log_model(
            model,
            name="model",
        )

    return model


def save_model(model, output_path):
    """Save trained model locally."""

    joblib.dump(
        model,
        output_path / "model.joblib",
    )


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]

    data_path = home_dir / input_file.lstrip("/")
    output_path = home_dir / "models"

    output_path.mkdir(parents=True, exist_ok=True)

    train_features = pd.read_csv(data_path / "train.csv")

    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    trained_model = find_best_model_with_params(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    save_model(
        trained_model,
        output_path,
    )


if __name__ == "__main__":
    main()
