from pathlib import Path
# import os
# import numpy as np
# import pandas as pd
# import logging
import json
from fastapi.testclient import TestClient  # import the TestClient class
from pydantic import ValidationError
from ..main import app, InputX  # , NumpyArray, NumpyEncoder, AnyJsonModel, ConstrainedJsonModel
from ..starter.train_model import load_data
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Instantiate the testing client with our app
client = TestClient(app)


# Write the unit tests using the same syntax as with the requests module.


def load_test_data():
    # load the data file specified by data_name and returns the data as a DataFrame
    data_df = load_data()
    data_df = data_df.drop('salary', axis=1)
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


def test_api_locally_post_input_data():
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


def test_api_locally_post_status():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :].to_dict()
    # json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert r.status_code == 200  # requests return 200 if successful


def test_api_locally_post_check_predict_type():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :].to_dict()
    # to json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert isinstance(r.json()["predict"], int)


def test_api_locally_post_check_malformed_1():
    # api call
    r = client.post("/inference", data='data')

    assert r.status_code != 200


def test_api_locally_post_check_malformed_2():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_dict = data_df.iloc[0, :].to_dict()
    # corrupt input by removing one required item
    x_dict.pop("age")
    # to json
    data = json.dumps({"json_obj": x_dict})
    try:
        # api call
        r = client.post("/inference", data=data)
        assert r.status_code != 200
        assert True, "ERROR: api call should have returned ValueError"
    except ValueError as e:
        assert True, "OK: api call returned ValueError as expected"


def test_api_locally_post_check_proba_type():
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[0, :].to_dict()
    # to json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert isinstance(r.json()["predict_proba_0"], float)


def test_api_locally_post_predict_0():
    ''' function should return 0 '''
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[8, :].to_dict()  # 8: sample with low proba0
    # json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert r.json()["predict"] == 1


def test_api_locally_post_predict_1():
    ''' function should return 1 '''
    # load test dataframe
    data_df = load_test_data()
    # select first data sample
    x_df = data_df.iloc[4, :].to_dict()  # 4: sample with high proba0
    # json
    data = json.dumps({"json_obj": x_df})
    # api call
    r = client.post("/inference", data=data)

    assert r.json()["predict"] == 0


if __name__ == '__main__':
    test_api_locally_get_root_status()
    print('OK - test_api_locally_get_root_status()\n')

    test_api_locally_get_root_response_type()
    print('OK - test_api_locally_get_root_response_type()\n')

    test_api_locally_get_root_response_msg()
    print('OK - test_api_locally_get_root_response_msg()\n')

    test_api_locally_post_input_data()
    print('OK - test_api_locally_post_input_data()\n')

    test_api_locally_post_status()
    print('OK - test_api_locally_post_status()\n')

    test_api_locally_post_check_predict_type()
    print('OK - test_api_locally_post_check_predict_type()\n')

    test_api_locally_post_check_proba_type()
    print('OK - test_api_locally_post_check_proba_type()\n')

    test_api_locally_post_predict_0()
    print('OK - test_api_locally_post_predict_0()\n')

    test_api_locally_post_predict_1()
    print('OK - test_api_locally_post_predict_1()\n')
