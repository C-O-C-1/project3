from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np


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
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)
    filename = 'trained_model.sav'
    joblib.dump(model_rf, filename)

    return model_rf, filename


def load_model(file_name):
    """
    Loads a trained model

    Inputs
    ------
    file_name : file name or path to model

    Returns
    -------
    loaded_model=loaded trained model
    """
    loaded_model = joblib.load(file_name)

    return loaded_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)
    return preds


def inference_categorical(X, encoder, lb, model):
    """ Run model inferences and return the the performances of the model on slices of the data.

    Inputs
    ------
    X : Test data used for prediction
    encoder: encoder for categorical features
    lb: label binarizer
    model : trained Random Forest model
    Returns
    -------
    preds : Predictions on data slices
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

    with open("slice_output.txt", "w") as file:
        for category in cat_features:
            file.write("-----" + '\n')
            file.write("CATEGORY " + category + '\n')
            file.write("-----" + '\n')
            print("-----")
            print("CATEGORY " + category)
            print("-----")
            for value in X[category].unique():
                file.write("-----" + '\n')
                file.write("VALUE " + value + '\n')
                file.write("-----" + '\n')
                print("-----")
                print("VALUE " + value)
                print("-----")
                sliced_df = X[X[category] == value]
                sliced_y = sliced_df["salary"]
                sliced_df = sliced_df.drop(["salary"], axis=1)
                X_categorical = sliced_df[cat_features].values
                X_continuous = sliced_df.drop(*[cat_features], axis=1)
                X_categorical = encoder.transform(X_categorical)
                y = lb.transform(sliced_y.values).ravel()
                X_test = np.concatenate([X_continuous, X_categorical], axis=1)
                preds = model.predict(X_test)
                fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
                precision = precision_score(y, preds, zero_division=1)
                recall = recall_score(y, preds, zero_division=1)
                file.write("-----Performance------" + '\n')
                file.write("F-BETA-SCORE " + str(fbeta) + '\n')
                file.write("Precision " + str(precision) + '\n')
                file.write("Recall " + str(recall) + '\n')
                print("-----Performance------")
                print("F-BETA-SCORE " + str(fbeta))
                print("Precision " + str(precision))
                print("Recall " + str(recall))

    file.close()
