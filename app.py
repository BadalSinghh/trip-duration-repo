import pathlib
import yaml
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel

from trip_duration.feature_definitions import feature_build


app = FastAPI(
    title="Trip Duration Prediction API",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    vendor_id: float
    pickup_datetime: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float


# -------------------------------------------------------------------
# Load trained model once when the API starts
# -------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Trip Duration Prediction API is running."}


@app.post("/predict")
def predict(input_data: PredictionInput):

    df = pd.DataFrame(
        [
            {
                "vendor_id": input_data.vendor_id,
                "pickup_datetime": input_data.pickup_datetime,
                "passenger_count": input_data.passenger_count,
                "pickup_longitude": input_data.pickup_longitude,
                "pickup_latitude": input_data.pickup_latitude,
                "dropoff_longitude": input_data.dropoff_longitude,
                "dropoff_latitude": input_data.dropoff_latitude,
                "store_and_fwd_flag": input_data.store_and_fwd_flag,
            }
        ]
    )

    with open(BASE_DIR / "params.yaml", "r") as f:
        params = yaml.safe_load(f)

    features = feature_build(
        df,
        "prediction",
        params,
    )

    prediction = float(model.predict(features)[0])

    return {"predicted_trip_duration": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", reload=True)
