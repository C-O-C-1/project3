import pytest
from ml import model
import train_model


@pytest.fixture
def dataset():
    """
    Fixture for loading the data

    Returns
    -------
    X_train: Data for training the model
    y_train: Labels of the train data
    X_test: Data for testing the model
    y_test: Labels of the test data
    encoder: encoder for categorical features
    lb: label binarizer
    """
    data_df = train_model.load_data("./data/census.csv")
    train, test = train_model.split_data(data_df)
    X_train, y_train, X_test, y_test, encoder, lb,  filename_enc, filename_lb = train_model.prepare_data(train, test)
    return X_train, y_train, X_test, y_test, encoder, lb,  filename_enc, filename_lb


def test_train_model(dataset):
    """
    Test if a model was trained by checking the target classes it was trained on
    Inputs
    X_train: Data for training the model
    y_train: Labels of the train data
    X_test: Data for testing the model
    y_test: Labels of the test data
    encoder: encoder for categorical features
    lb: label binarizer
    ------

    Returns
    None
    -------

    """
    X_train, y_train, X_test, y_test, encoder, lb,  filename_enc, filename_lb = dataset
    model_rf, file_model = model.train_model(X_train, y_train)
    assert len(model_rf.classes_) > 0


def test_inference(dataset):
    """
    Test if inference is run correctly
    Inputs
    ------
    X_train: Data for training the model
    y_train: Labels of the train data
    X_test: Data for testing the model
    y_test: Labels of the test data
    encoder: encoder for categorical features
    lb: label binarizer

    Returns
    -------
    None
    """
    X_train, y_train, X_test, y_test, encoder, lb,  filename_enc, filename_lb = dataset
    model_rf, file_model = model.train_model(X_train, y_train)
    preds = preds = model.inference(model_rf, X_test)
    assert len(preds) == len(X_test)


def test_compute_model_metrics(dataset):
    """
    Test if metrics are computed and type is correct (float or int between 0 and 1)
    Inputs
    ------
    X_train: Data for training the model
    y_train: Labels of the train data
    X_test: Data for testing the model
    y_test: Labels of the test data
    encoder: encoder for categorical features
    lb: label binarizer

    Returns
    -------
    None
    """
    X_train, y_train, X_test, y_test, encoder, lb,  filename_enc, filename_lb = dataset
    model_rf, file_model = model.train_model(X_train, y_train)
    preds = model_rf.predict(X_test)
    precision, recall, fbeta = model.compute_model_metrics(y_test, preds)
    assert type(precision) is int or float
    assert type(recall) is int or float
    assert type(fbeta) is int or float
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
