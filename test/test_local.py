from fastapi.testclient import TestClient   # import the TestClient class
from ..starter.main import app

# Instantiate the testing client with our app
client = TestClient(app)

# Write the unit tests using the same syntax as with the requests module.


def test_api_locally_get_root():  # name of current unit test function
    r = client.get("/")   # a request against the root domain
    assert r.status_code == 200   # requests return 200 if successful
#    assert isinstance(r.json, str)

def test_api_locally_inference():  # name of current unit test function
    r = client.post("/udacity_project_4")
    assert r.status_code == 200   # requests return 200 if successful
#    assert isinstance(r.json, str)
