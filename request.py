import pandas as pd
import requests
import train_model
from typing import List, Dict, Any

data_path: str = "./data/census.csv"


def load_data(pth: str) -> pd.DataFrame:
    """Loads data

    Inputs
    ------
    pth : path to data

    Returns
    -------
    data_df: Pandas Dataframe with data
    """
    data_df: pd.DataFrame = train_model.load_data(pth)
    return data_df


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    """Splits data

    Inputs
    ------
    df : Pandas dataframe with data

    Returns
    -------
    test: Pandas dataframe containing test data
    """
    train, test = train_model.split_data(df)
    return test


def prepare_data(df_test: pd.DataFrame) -> None:
    """
    Manages request for model inference with fast API app
    Prints model prediction
    Inputs
    ------
    df_test : Pandas dataframe with test data

    Returns
    -------
    test_data: dict with dict of lists
    """
    test_dict: List[Dict[str, Any]] = df_test.to_dict("records")
    test_data: Dict[str, List[Dict[str, Any]]] = {"data": test_dict}

    return test_data


if __name__ == "__main__":
    data_path: str = "./data/census.csv"
    loaded_data: pd.DataFrame = load_data(data_path)
    test_data: pd.DataFrame = split_data(loaded_data)
    prepared_test_data: Dict[str, List[Dict[str, Any]]] = prepare_data(test_data)
    response = requests.post("http://127.0.0.1:8000/inference", json=prepared_test_data)
    print(response.status_code)
    print(response.text)
