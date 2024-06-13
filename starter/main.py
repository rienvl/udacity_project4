''' main.py - contains the definition of the API '''
from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=True).parent
# import os
# import numpy as np
# import json
# import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, Json
from typing import Dict  # , Any, List
import logging
from starter.ml.data import process_data
from starter.ml.model import load_model, inference
from starter.train_model import clean_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


class InputX(BaseModel):
    json_obj: Dict

    model_config = {
        "json_schema_extra": {
            "examples": [{
                    "json_obj": {
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
                        "native-country": "United- States"
                    }}]}}


def convert_json(x_dict):
    # print("convert_json() started")
    # print(isinstance(x_dict, dict))  # X.json_obj is a dict
    # print(X.json_obj.keys())
    # convert dict to dataframe
    data = pd.DataFrame(x_dict, index=[0])

    # remove rows with missing data
    data = clean_data(data)
    # logging.info("OK - convert_json(): cleaned test data")

    # load model
    model, encoder, lb = load_model()
    # logging.info("OK - convert_json(): loaded model items")

    # proces the test data with the process_data function
    X_test, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # logging.info("OK - convert_json(): processed data, {} rows".format(X_test.shape[0]))
    # print("convert_json() finished")

    return model, X_test


app = FastAPI()  # instantiate the app


@app.get("/")
async def get_root():
    return {"greetings": "welcome to project_4", 'status_code': 200}


# a POST that does model inference
@app.post("/inference")    # defines as endpoint: http://localhost:8000/inference
async def post_model_inference(data: InputX):
    model, x_data = convert_json(data.json_obj)
    predict, predict_proba = inference(model, x_data)
    print("predict = {},  predict_proba[0] = {},  predict_proba[1] = {}"
          .format(predict[0], predict_proba[0, 0], predict_proba[0, 1]))

    return {'predict': int(predict[0]), 'predict_proba_0': float(predict_proba[0, 0])}


# class NumpyArray(BaseModel):
#     X: np.ndarray = Field(default_factory=lambda: np.zeros(10))
#
#     class Config:
#         arbitrary_types_allowed = True
#
#     # model_config = {
#     #     "json_schema_extra": {
#     #         "examples": [{
#     #             "X": np.zeros(200)
#     #         }]}}
#
#
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
#
#
# class AnyJsonModel(BaseModel):
#     json_obj: Json[Any]
#
#
# class ConstrainedJsonModel(BaseModel):
#     json_obj: Json[List[float]]
