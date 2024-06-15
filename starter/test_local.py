from pathlib import Path
# import os
# import numpy as np
# import pandas as pd
# import logging
import json
from fastapi.testclient import TestClient  # import the TestClient class
from pydantic import ValidationError
from .main import app, InputX  # , NumpyArray, NumpyEncoder, AnyJsonModel, ConstrainedJsonModel
from .starter.train_model import load_data
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Instantiate the testing client with our app
client = TestClient(app)


# Write the unit tests using the same syntax as with the requests module.


def load_test_data():
    # load the data file specified by data_name and returns the data as a DataFrame
    data_df = load_data()
    data_df = data_df.drop('salary', axis=1)
    print(data_df.columns)

    # logging.info("OK - load_test_data(): loaded test dataset")

    return data_df


def test_api_locally_get_root_status():
    r = client.get("/")  # a request against the root domain
    assert r.status_code == 200  # requests return 200 if successful


def test_api_locally_get_root_response_type():
    r = client.get("/")  # a request against the root domain
    assert isinstance(r.json()["greetings"], str)


def test_api_locally_get_root_response_msg():
    r = client.get("/")  # a request against the root domain
    assert r.json()["greetings"] == "welcome to project_4"


def test_api_locally_inference_input_data():
    ''' test if input is valid json object '''
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :]
    print(x_df)
    try:
        InputX(json_obj=x_df.to_dict())
        print("OK - input x_df is a valid json object")
    except ValidationError as e:
        print("ERROR - input x_df is not a valid json object")


def test_api_locally_inference_status():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :].to_dict()  # for 2d: use orient='split'?
    # json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert r.status_code == 200  # requests return 200 if successful


def test_api_locally_inference_check_predict_type():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :].to_dict()  # for 2d: use orient='split'?
    # to json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert isinstance(r.json()["predict"], int)


def test_api_locally_inference_check_proba_type():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :].to_dict()  # for 2d: use orient='split'?
    # to json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert isinstance(r.json()["predict_proba_0"], float)


if __name__ == '__main__':
    test_api_locally_get_root_status()
    print('OK - test_api_locally_get_root_status()\n')

    test_api_locally_get_root_response_type()
    print('OK - test_api_locally_get_root_response_type()\n')

    test_api_locally_get_root_response_msg()
    print('OK - test_api_locally_get_root_response_msg()\n')

    test_api_locally_inference_input_data()
    print('OK - test_api_locally_inference_input_data()\n')

    test_api_locally_inference_status()
    print('OK - test_api_locally_inference_status()\n')

    test_api_locally_inference_check_predict_type()
    print('OK - test_api_locally_inference_check_predict_type()\n')

    test_api_locally_inference_check_proba_type()
    print('OK - test_api_locally_inference_check_proba_type()\n')
