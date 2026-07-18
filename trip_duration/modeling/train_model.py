import pathlib
import sys
import joblib
import yaml
import mlflow
import mlflow.xgboost
import pandas as pd

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


def find_best_model_with_params(
    X_train,
    y_train,
    X_test,
    y_test,
    params,
):
    """
    Perform Hyperopt tuning with MLflow tracking.
    """

    xgb_params = params["xgboost"]["search_space"]

    space = {
        "n_estimators": hp.choice(
            "n_estimators",
            xgb_params["n_estimators"],
        ),
        "max_depth": hp.choice(
            "max_depth",
            xgb_params["max_depth"],
        ),
        "learning_rate": hp.uniform(
            "learning_rate",
            xgb_params["learning_rate"]["min"],
            xgb_params["learning_rate"]["max"],
        ),
    }

    def evaluate_model(hyper_params):

        hyper_params["max_depth"] = int(hyper_params["max_depth"])

        with mlflow.start_run(nested=True):
            mlflow.log_params(hyper_params)

            model = XGBRegressor(
                random_state=params["split"]["random_state"],
                **hyper_params,
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test,
            )

            rmse = root_mean_squared_error(
                y_test,
                predictions,
            )

            mlflow.log_metric(
                "RMSE",
                rmse,
            )

            return {
                "loss": rmse,
                "status": STATUS_OK,
            }

    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=params["mlflow"]["tuning_run_name"]):
        trials = Trials()

        best = fmin(
            fn=evaluate_model,
            space=space,
            algo=tpe.suggest,
            max_evals=params["hyperopt"]["max_evals"],
            trials=trials,
        )

        best_params = space_eval(
            space,
            best,
        )

        best_params["max_depth"] = int(best_params["max_depth"])

        mlflow.log_params(best_params)

        model = XGBRegressor(
            random_state=params["split"]["random_state"],
            **best_params,
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test,
        )

        rmse = root_mean_squared_error(
            y_test,
            predictions,
        )

        mlflow.log_metric(
            "Final RMSE",
            rmse,
        )

        mlflow.xgboost.log_model(
            model,
            name="model",
        )

    return model


def save_model(
    model,
    output_path,
):
    """
    Save trained model locally.
    """

    joblib.dump(
        model,
        output_path / "model.joblib",
    )


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    with open(
        home_dir / "params.yaml",
        "r",
    ) as f:
        params = yaml.safe_load(f)

    input_file = sys.argv[1]

    data_path = home_dir / input_file.lstrip("/")

    output_path = home_dir / "models"
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = params["data"]["target"]

    train_features = pd.read_csv(data_path / "train.csv")

    X = train_features.drop(columns=[target])

    y = train_features[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_state"],
    )

    trained_model = find_best_model_with_params(
        X_train,
        y_train,
        X_test,
        y_test,
        params,
    )

    save_model(
        trained_model,
        output_path,
    )


if __name__ == "__main__":
    main()
