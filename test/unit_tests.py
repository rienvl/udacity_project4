import os
import pytest
import pandas as pd
import numpy as np
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
def model_encoder_lb():
    try:
        model,  encoder, lb = mdl.load_model()
        logging.info("OK - pytest.py: loaded model, encoder, and lb")
    except:
        logging.info("ERROR - unit_tests.py: model loading returned error")
        pytest.fail("fixture model_encoder()")

    return model, encoder, lb


def test_load_data(test_data):
    """check if test data has expected number of rows"""
    assert test_data.shape[0] == 32561, 'test_data has incorrect number of rows'


def test_clean_data(clean_test_data):
    """check if clean data step returns the correct number of rows"""
    assert clean_test_data.shape[0] == 30162, 'clean_data() returned incorrect number of rows'


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
        assert predict.shape[0] == X_train.shape[0], 'train_model() returned output with wrong shape'
    except:
        pytest.fail("test_train_model")


def test_inference(model_encoder_lb, clean_test_data):
    model = model_encoder_lb[0]
    encoder = model_encoder_lb[1]
    lb = model_encoder_lb[2]
    X_test, _, _, _ = process_data(
        clean_test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logging.info("OK - test_inference.py: processed data")

    predict = mdl.inference(model, X_test)
    assert isinstance(predict[0], np.int64), 'inference() returned wrong type'
    assert (predict.shape[0] == X_test.shape[0]), 'inference() returned output with wrong shape'


def test_compute_model_metrics(model_encoder_lb, clean_test_data):
    model = model_encoder_lb[0]
    encoder = model_encoder_lb[1]
    lb = model_encoder_lb[2]
    X_test, y_test, _, _ = process_data(
        clean_test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logging.info("OK - test_compute_model_metrics.py: processed data")

    predict = mdl.inference(model, X_test)

    f1_score = -1
    precision = -1
    recall = -1
    try:
        f1_score, precision, recall = mdl.compute_model_metrics(y_test, predict)
    except:
        logging.info("ERROR - unit_tests.py: mdl.compute_model_metrics() returned error")
        pytest.fail("test_compute_model_metrics()")

    assert isinstance(f1_score, float), 'compute_model_metrics() returned wrong type for f1_score'
    assert isinstance(precision, float), 'compute_model_metrics() returned wrong type for precision'
    assert isinstance(recall, float), 'compute_model_metrics() returned wrong type for recall'
    assert (0 < f1_score < 1.0), 'compute_model_metrics() returned unrealistic value for f1_score'
    assert (0 < precision < 1.0), 'compute_model_metrics() returned unrealistic value for precision'
    assert (0 < recall < 1.0), 'compute_model_metrics() returned unrealistic value for recall'


# def test_slice_averages(clean_test_data):
#     """ Test to see if the mean per categorical slice is in the range 1.5 to 2.5 """
#     for cat_feat in cat_features:
#         avg_value = clean_test_data[clean_test_data[cat_feat] == cat_feat]["numeric_feat"].mean()
#         assert (
#             2.5 > avg_value > 1.5
#         ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."
