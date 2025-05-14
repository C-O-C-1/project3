from fastapi.testclient import TestClient
from main import app
import request


client = TestClient(app)


def test_welcome_user():
    """
    Tests status code and contents of the request.

    Returns
    -------
    None
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"welcome": "Welcome to our classification model"}


def test_make_prediction_class_0():
    """
    Tests that model predicts class 0

    Returns
    -------
    None
    """
    data_path = "./data/census.csv"
    data_df = request.load_data(data_path)
    test_data = request.split_data(data_df)
    prep_test_data = request.prepare_data(test_data)
    response = client.post("/inference", json=prep_test_data)
    assert response.status_code == 200
    assert any(pred in [0.0] for pred in response.json())


def test_make_prediction_class_1():
    """
    Tests that model predicts class 1

    Returns
    -------
    None
    """
    data_path = "./data/census.csv"
    data_df = request.load_data(data_path)
    test_data = request.split_data(data_df)
    prep_test_data = request.prepare_data(test_data)
    response = client.post("http://127.0.0.1:8000/inference", json=prep_test_data)
    assert response.status_code == 200
    assert any(pred in [1.0] for pred in response.json())
