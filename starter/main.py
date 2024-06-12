''' main.py - contains the definition of the API '''
from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=True).parent
import os
from fastapi import FastAPI
import logging
import numpy as np
from sklearn.model_selection import train_test_split
# from starter.starter.ml.data import process_data
# from starter.starter.ml.model import load_model, inference
# from starter.starter.train_model import load_data, clean_data
from starter.ml.data import process_data
from starter.ml.model import load_model, inference
from starter.train_model import load_data, clean_data
import json


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_model_and_data():
    logging.info("OK - current path = {}".format(os.getcwd()))
    # load the data file specified by data_name and returns the data as a DataFrame
    data = load_data()
    # remove rows with missing data
    data = clean_data(data)
    # split data
    train_data, test_data = train_test_split(data, test_size=0.20)
    logging.info("OK - train_model.py: train-test split completed")

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

    # load model
    logging.info("OK - current path = {}".format(os.getcwd()))
    model, encoder, lb = load_model()

    # proces the test data with the process_data function
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logging.info("OK - train_model.py: processed test  data, {} rows".format(y_test.shape[0]))

    return model, X_test


# class NumpyArray(BaseModel):
#     numpyArray: np.ndarray #= Field(default_factory=lambda: np.zeros(10))
#     class Config:
#         arbitrary_types_allowed = True


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = FastAPI()  # instantiate the app


@app.get("/")      # “/” : defines the default endpoint location http://127.0.0.1:8000
async def get_root():
    return {"greeting": "welcome to project_4", 'status': 200}  # return a JSON response on the browser (URL: see below)


# a POST that does model inference
@app.post("/inference")    # defines as endpoint: http://localhost:8000/inference
async def post_model_inference():
    model, X_test = load_model_and_data()
    print(X_test.shape)
    predicts = inference(model, X_test)
    prediction_list = json.dumps({'predicts': predicts}, cls=NumpyEncoder)

    return {'predictions_list': prediction_list, 'status': 200}  # add return value for prediction outputs


# @app.get("/items/{item_id}")
# async def get_items(item_id: int, count: int = 1):
#     return {"fetch": f"Fetched {count} of {item_id}"}
