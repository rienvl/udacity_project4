# Link to Github Repository:
https://github.com/rienvl/udacity_project4

## Project 4 - deploying a ML model to cloud application platform


# Environment Set up
use:
* pip install -r requirements.txt using python 3.10.14
or use:
* conda create -n [envname] "python=3.10.14" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
* python 3.10.14 is required by Heroku


# Model
The purpose of the model is to estimate if the income of an individual is below or above 50k.
The model is a standard logistic regression model from sklearn ("LogisticRegression"). 
Parameter C (: the inverse of regularization strength) was set to 1.0.
The model was trained on the UCI Census Income Dataset; a dataset meant for research purposes.

##Intended Use
The dataset is purely for research purposes and of poor quality.
The effort put in model training is therefore limited and no conclusions should be drawn from its output.  

# Training Data
Input data contains the following features:
6 continuous  features: {age, capital-gain, capital-loss, education-num, fnlgt, hours-per-week}
8 categorical features: {workclass, education, marital-status, occupation, relationship, race, sex, native-country}
And class label "salary".
Using a 80/20% train/test split resulted in 24129 valid training samples.

# GitHub Actions

* pytest consists of 2 parts:
1) unit tests (/test/unit_test.py)
2) tests for api (test_local_api.py)
* besides pytest we test flake8

