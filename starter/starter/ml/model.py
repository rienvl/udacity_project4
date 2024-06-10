from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


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
    precision : float
    recall : float
    f_beta : float
    """
    f_beta = fbeta_score(y, predict, beta=1, zero_division=1)
    precision = precision_score(y, predict, zero_division=1)
    recall = recall_score(y, predict, zero_division=1)
    return precision, recall, f_beta


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
    predict = model.predict(X)  # .reshape(-1, 1))
    return predict
