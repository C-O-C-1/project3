from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from ml import data, model
from typing import List

app = FastAPI()


@app.get("/")
async def welcome_user() -> dict[str, str]:
    """
    Returns greeting message

    Returns
    -------
    Greeting message
    """
    return {"welcome": "Welcome to our classification model"}


class InferenceData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "salary": "<=50K"
            }
        }
        allow_population_by_alias = True


class TotalInferenceData(BaseModel):
    data: list[InferenceData]


@app.post("/inference/", response_model=List[float])
async def make_prediction(datapoints: TotalInferenceData) -> List[float]:
    """
    Manages model prediction
    Inputs
    ------
    datapoints : Object containing the data

    Returns
    -------
    list with the model predictions
    """
    test_datapoints: List = datapoints.data
    test_data_df: pd.DataFrame = pd.DataFrame([item.dict() for item in test_datapoints])
    loaded_model = model.load_model("trained_model.sav")
    encoder, lb = data.load_encoder_lb("encoder_cat.sav", "lb.sav")
    y: pd.DataFrame = test_data_df["salary"]
    test_data_df: pd.DataFrame = test_data_df.drop(["salary"], axis=1)
    cat_features: List[str] = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]
    X_categorical: np.array = test_data_df[cat_features].values
    X_continuous: pd.DataFrame = test_data_df.drop(*[cat_features], axis=1)
    X_categorical: np.array = encoder.transform(X_categorical)
    y: np.array = lb.transform(y.values).ravel()
    X: np.array = np.concatenate([X_continuous, X_categorical], axis=1)
    predictions: np.array = loaded_model.predict(X)
    predictions_lst: List[float] = [float(pred) for pred in predictions]
    return predictions_lst
