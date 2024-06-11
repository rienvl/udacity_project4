import os
import numpy as np
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               # multi_class='warn', n_jobs=None, penalty='l2',
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    model = logit.fit(X_train, y_train)
    return model


def load_model(parent_path='~/git/udacity_project_4/'):
    '''
    this function loads the model and encoder specified by inputs

    Returns model, encoder, lb
    -------
    '''
    full_folder_path = os.path.join(parent_path, 'starter', 'model')
    full_path = os.path.join(full_folder_path, 'trainedmodel.pkl')
    # load model
    if not os.path.isfile(full_path):
        print('WARNING Is not file: {}'.format(full_path))

    with open(full_path, 'rb') as file:
        model = pickle.load(file)
    # load encoder
    full_path = os.path.join(full_folder_path, 'trainedencoder.pkl')
    print(os.path.isfile(full_path))
    with open(full_path, 'rb') as file:
        encoder = pickle.load(file)
    # load encoder
    full_path = os.path.join(full_folder_path, 'trainedlb.pkl')
    print(os.path.isfile(full_path))
    with open(full_path, 'rb') as file:
        lb = pickle.load(file)
    logging.info("OK - model.py: loaded model, encoder, and lb")

    return model, encoder, lb


def save_model(model, encoder, lb, parent_path='~/git/udacity_project_4/'):
    '''
    this function saves the input model and encoder as pickle files
    Parameters
    ----------
    model: input trained model
    # model_name: name of the output pkl file
    encoder: input trained encoder
    # encoder_name: name of the output pkl file
    lb: trained label binarizer
    -------
    '''
    full_folder_path = os.path.join(parent_path, 'starter', 'model')
    # save model
    full_path = os.path.join(full_folder_path, 'trainedmodel.pkl')
    filehandler = open(full_path, 'wb')
    pickle.dump(model, filehandler)
    logging.info("OK - model.py: stored model           in {}".format(full_path))
    # save encoder
    full_path = os.path.join(full_folder_path, 'trainedencoder.pkl')
    filehandler = open(full_path, 'wb')
    pickle.dump(encoder, filehandler)
    logging.info("OK - model.py: stored encoder         in {}".format(full_path))
    # save lb
    full_path = os.path.join(full_folder_path, 'trainedlb.pkl')
    filehandler = open(full_path, 'wb')
    pickle.dump(lb, filehandler)
    logging.info("OK - model.py: stored label binarizer in {}".format(full_path))


def compute_model_metrics(y, predict):
    """
    Validates the trained machine learning model using precision, recall, and f_beta.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    predict : np.array
        Predicted labels, binarized.
    Returns
    -------
    f1_score : float
    precision : float
    recall : float
    """
    f1 = f1_score(y, predict, zero_division=np.NaN)
    precision = precision_score(y, predict, zero_division=1)
    recall = recall_score(y, predict, zero_division=1)
    return f1, precision, recall


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn logistic regression model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    predict : np.array
        Predictions from the model.
    """
    predict = model.predict(X)
    return predict


def get_model_performance_on_slices(X_val, y_val, cat_features, model, encoder):
    slice_cat = encoder.get_feature_names_out(cat_features)
    # X_val.shape[1] := n_continuous + n_categorical
    n_continuous = X_val.shape[1] - len(slice_cat)
    print('n_continuous = {},   n_categorical = {}'.format(n_continuous, len(slice_cat)))
    f1_score_list = []
    for idx, column in enumerate(slice_cat):
        column_idx = n_continuous + idx
        f1 = np.NaN  # initialize as nan
        if np.count_nonzero(X_val[:, column_idx]) > 10:
            # only compute f1 score if sufficient samples
            y_val_sub = y_val[X_val[:, column_idx] > 0]
            X_val_sub = X_val[X_val[:, column_idx] > 0]
            pred_sub = model.predict(X_val_sub)
            f1 = f1_score(y_val_sub, pred_sub, zero_division=np.NaN)
            print('F1 score {:.3f} for slices {}'.format(f1, column))
        else:
            print('no reliable score: {}'.format(column))

        f1_score_list.append(f1)

    return f1_score_list, slice_cat


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ..train_model import clean_data
    from .data import process_data

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

    # load trained model
    model, encoder, lb = load_model()
    # load test data
    full_input_path = os.path.join('starter', 'data', 'census.csv')
    data = pd.read_csv(full_input_path)
    # clean test data
    data = clean_data(data)
    # split data
    train_data, test_data = train_test_split(data, test_size=0.20)
    # proces the test data
    X_test, y_test, _, _ = process_data(
        test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # get model performance on slices
    get_model_performance_on_slices(X_test, y_test, cat_features, model, encoder)
