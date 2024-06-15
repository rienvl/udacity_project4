import requests
import json
from starter.main import InputX


x_dict = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United- States"
    }

data = {"json_obj": x_dict}

# request against local:
# response = requests.post('http://0.0.0.0:8000/inference', data=json.dumps(data))

# request against Heroku live application
response = requests.post('https://udacity-project-4-app-3bfaebc2749a.herokuapp.com/inference', data=json.dumps(data))
print(response.status_code)
print(response.json())