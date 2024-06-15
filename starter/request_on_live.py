import requests
import json
from main import InputX


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

# response = requests.post('/inference')
# response = requests.post('https://udacity-project-4-app-3bfaebc2749a.herokuapp.com/')
#response = requests.post('http://127.0.0.1:8000/inference')
#response = requests.post('http://localhost:8000/inference', data=json.dumps(data))
response = requests.post('http://0.0.0.0:8000/inference', data=json.dumps(data))
#response = requests.post('/url/to/query/', auth=('usr', 'pass'), data=json.dumps(data))
print(response.status_code)
print(response.json())