''' main.py - contains the definition of the API '''
import os
from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel, Field
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import load_model, inference
from starter.train_model import load_data, clean_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_model_and_data(parent_path='~/git/udacity_project_4'):
    logging.info("OK - current path = {}".format(os.getcwd()))
    # load the data file specified by data_name and returns the data as a DataFrame
    data = load_data(parent_path)
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
    model, encoder, lb = load_model(parent_path)

    # proces the test data with the process_data function
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logging.info("OK - train_model.py: processed test  data, {} rows".format(y_test.shape[0]))

    return model, X_test


class NumpyArray(BaseModel):
    numpyArray: np.ndarray #= Field(default_factory=lambda: np.zeros(10))
    class Config:
        arbitrary_types_allowed = True

class TaggedItem(BaseModel):
    name: str
    tags: Union[str, List[str]]
    item_id: int


app = FastAPI()  # instantiate the app


@app.get("/")      # “/” : defines the default endpoint location http://127.0.0.1:8000
async def get_root():
    return {"greeting": "welcome to project_4"}  # return a JSON response on the browser (URL: see below)


# a POST that does model inference
@app.post("/inference")    # defines as endpoint: http://localhost:8000/inference
async def post_model_inference():
    parent_path = '~/git/udacity_project_4'
    model, X_test = load_model_and_data(parent_path)
    print(X_test.shape)
    predictions_list = inference(model, X_test)

    return {'predictions_list': predictions_list, 'status': 200}  # add return value for prediction outputs


# @app.get("/items/{item_id}")
# async def get_items(item_id: int, count: int = 1):
#     return {"fetch": f"Fetched {count} of {item_id}"}


if __name__ == '__main__':
    os.system('uvicorn main:app --reload')
