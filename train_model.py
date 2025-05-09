# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml import data, model
import pandas as pd

data_path = "./data/census.csv"


def load_data(path):
    """ Loads .csv dataset into pandas dataframe

    Inputs
    ------
    path: path to dataset

    Returns
    -------
    data : clean pandas dataframe
    """

    data = pd.read_csv(path)
    data.rename(columns={cl: cl.strip() for cl in data.columns}, inplace=True)

    return data


def split_data(df):
    """Split data into train/test sets

    Inputs
    ------
    df: pandas Dataframe with data

    Returns
    -------
    train : train subset
    tes: test subset
    """
    train, test = train_test_split(df, test_size=0.20)
    return train, test


def prepare_data(train, test):
    """Transforms data for models

    Inputs
    ------
    train : train subset
    tes: test subset

    Returns
    -------
    X_train: datapoints of train data
    y_train: labels of train data
    X_test: datapoints of test data
    y_test: labels of test data
    encoder: categorical encoder
    lb: label binarizer
    filename_enc: path to sabed encoder
    filename_lb: path to saved lb
    """
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
    X_train, y_train, encoder, lb, filename_enc, filename_lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb, filename_enc, filename_lb = data.process_data(
        test, categorical_features=cat_features,
        encoder=encoder, lb=lb, label="salary", training=False
    )
    return X_train, y_train, X_test, y_test, encoder, lb, filename_enc, filename_lb


if __name__ == "__main__":
    # Train and save a model.
    data_df = load_data(data_path)
    train, test = split_data(data_df)
    X_train, y_train, X_test, y_test, encoder, lb, filename_enc, filename_lb = prepare_data(train, test)
    rf_model, filename = model.train_model(X_train, y_train)
    loaded_model = model.load_model(filename)
    loaded_enc, loaded_lb = data.load_encoder_lb(filename_enc, filename_lb)
    preds = model.inference(loaded_model, X_test)
    preds_sliced = model.inference_categorical(test, loaded_enc,
                                               loaded_lb, loaded_model)
