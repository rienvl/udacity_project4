import os
import pytest
import pandas as pd
import pickle
import logging
from ..starter.starter.ml import model as mdl
from ..starter.starter.ml.data import process_data
from ..starter.starter.train_model import clean_data


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


@pytest.fixture(scope="module")
def test_data():
    full_input_path = os.path.join('starter', 'data', 'census.csv')
    data = None
    try:
        data = pd.read_csv(full_input_path)
    except FileNotFoundError:
        logging.info("ERROR - unit_tests.py: file not found")
        pytest.fail("fixture test_data()")

    if data is None:
        logging.info("ERROR - unit_tests.py: csv read of test data returned NaN")
        pytest.fail("fixture test_data()")

    return data


@pytest.fixture(scope="module")
def clean_test_data(test_data):
    try:
        test_data = clean_data(test_data)
    except:
        logging.info("ERROR - unit_tests.py: clean_data() step returned error")
        pytest.fail("fixture clean_test_data()")

    return test_data


@pytest.fixture(scope="module")
def model():
    try:
        full_model_path = os.path.join(os.getcwd(), './starter', 'model', 'trainedmodel.pkl')
        with open(full_model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("OK - pytest.py: loaded model")
    except:
        logging.info("ERROR - unit_tests.py: model loading returned error")
        pytest.fail("fixture model()")

    return model


def test_load_data(test_data):
    """check if test data has expected number of rows"""
    assert test_data.shape[0] == 32561


def test_clean_data(clean_test_data):
    """check if clean data step returns the correct number of rows"""
    assert clean_test_data.shape[0] == 30162


def test_train_model(clean_test_data):
    try:
        # Proces the train data with the process_data function
        X_train, y_train, encoder, lb = process_data(
            clean_test_data, categorical_features=cat_features, label="salary", training=True
        )
        logging.info("OK - test_train_model(): processed train data, {} rows".format(y_train.shape[0]))
        model = mdl.train_model(X_train, y_train)
        logging.info("OK - test_train_model(): model training completed")
        # test if valid model that returns predictions
        predict = model.predict(X_train)
        assert predict.shape[0] == X_train.shape[0]
    except:
        pytest.fail("test_train_model")


def test_inference(model, clean_test_data):
    X_train, y_train, _, _ = process_data(
        clean_test_data, categorical_features=cat_features, label="salary", training=True
    )
    logging.info("OK - test_inference.py: processed data")

    predict = mdl.inference(model, X_train)

    assert (predict.shape[0] == X_train.shape[0])


def test_compute_model_metrics(model, clean_test_data):
    X_train, y_train, _, _ = process_data(
        clean_test_data, categorical_features=cat_features, label="salary", training=True
    )
    logging.info("OK - test_compute_model_metrics.py: processed data")

    predict = mdl.inference(model, X_train)

    precision = -1
    recall = -1
    f_beta = -1
    try:
        precision, recall, f_beta = mdl.compute_model_metrics(y_train, predict)
    except:
        logging.info("ERROR - unit_tests.py: mdl.compute_model_metrics() returned error")
        pytest.fail("test_compute_model_metrics()")

    assert (0 < precision < 1.0)
    assert (0 < recall < 1.0)
    assert (0 < f_beta < 1.0)
