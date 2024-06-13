from pathlib import Path
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml import model as mdl
BASE_DIR = Path(__file__).resolve(strict=True).parent

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_data():
    '''
    this function loads the data file specified by data_name and returns the data as a DataFrame
    Parameters
    ----------
    data_name: name of the csv data file stored in folder /starter/data
    Returns data: pandas DataFrame
    -------
    '''
    # read in data using the pandas module
    full_path = Path(BASE_DIR).joinpath(f"../data/census.csv")
    data = pd.read_csv(full_path)
    logging.info("OK - train_model.py: loaded training data containing {} rows".format(data.shape[0]))
    return data


def clean_data(data_df):
    '''
    this function replaces "?" data with nans and then removes
    rows with any nan values from the input dataframe
    Parameters
    ----------
    data_df: input dataframe
    Returns data_df: cleaned dataframe
    -------
    '''
    # 2-steps: (1) replace "?" by nan, (2) remove rows with nans
    n_rows_orig = data_df.shape[0]
    data_df = data_df.replace('?', np.nan)
    data_df = data_df.dropna()
    n_rows = data_df.shape[0]
    percentage_missing_data = int(100 * float(n_rows_orig-n_rows) / (float(n_rows_orig)) + 0.5)
    logging.info("OK - train_model.py: removed {:d}% missing data".format(percentage_missing_data))

    return data_df


def train_model():
    '''
    this function
    1) loads the raw data specified by input data_name
    2) splits the data in train_data and test_data using test_size=0.20
    3) processes the train data
    3) processes the test  data

    Parameters
    ----------
    data_name: name of the csv data file stored in folder /starter/data

    # Returns: trained model
    -------
    '''
    # load the data file specified by data_name and returns the data as a DataFrame
    data = load_data()

    # remove rows with missing data
    data = clean_data(data)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    test_size = 0.20
    train_data, test_data = train_test_split(data, test_size=test_size)
    logging.info("OK - train_model.py: train-test split using test_size={}".format(test_size))

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

    # Proces the train data with the process_data function
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=cat_features, label="salary", training=True
    )
    logging.info("OK - train_model.py: processed train data, {} rows".format(y_train.shape[0]))

    # Proces the test data with the process_data function
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logging.info("OK - train_model.py: processed test  data, {} rows".format(y_test.shape[0]))

    # train model
    model = mdl.train_model(X_train, y_train)
    logging.info("OK - train_model.py: model training completed")

    # model inferences
    predictions = mdl.inference(model, X_test)
    logging.info("OK - train_model.py - inference completed")

    # validates the trained machine learning model using precision, recall, and f_beta
    f1, precision, recall = mdl.compute_model_metrics(y_test, predictions)
    logging.info("OK - train_model.py - validation completed:\n                             "
                 "f1_score = {:.4f},   precision = {:.4f},   recall = {:.4f}".format(f1, precision, recall))

    # save a model, encoder, and lb
    mdl.save_model(model=model, encoder=encoder, lb=lb)


if __name__ == '__main__':
    train_model()
