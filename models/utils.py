import os
import pprint
from datetime import datetime

import numpy as np
import pandas as pd
from comet_ml import Experiment
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

load_dotenv()
RANDOM_SEED = int(os.environ.get("RANDOM_SEED"))

SAVE_PREDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "preds")
ORIG_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orig")
PROC_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def save_preds(preds: pd.Series, model_name: str, params: dict, metrics: dict) -> None:
    """Saves the predictions of a model with the expected format by Kaggle under data/preds/model_name.
    Also logs the hyperparams and relevent metrics in a similarly named file with .txt extension.

    Args:
        preds (pd.Series): Predictions of the model. Not one-hot-enoded
        model_name (str): Name of the model. Normally passed on as the name of the calling file
        params (dict): hyper-parameters of the model. 
        metrics (dict): Relevent metrics of the model (accuracy, f1, recall, precision)
    """
    df = pd.DataFrame(range(len(preds)), columns=["S.No"])
    df["LABELS"] = preds.astype(int)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M")
    counts = np.unique(preds, return_counts=True)
    if counts[1].ndim == 2:
        print(f"Class Counts: \n0: {counts[1][0]} \t1: {counts[1][1]}")
    if not os.path.exists(os.path.join(SAVE_PREDS_PATH, model_name)):
        os.mkdir(os.path.join(SAVE_PREDS_PATH, model_name))
    f_name = os.path.join(SAVE_PREDS_PATH, model_name, f"{dt_string}({model_name})")
    df.to_csv(f"{f_name}.csv", index=False)
    with open(f"{f_name}.txt", "w") as f:
        pprint.pprint(metrics, f)
        pprint.pprint(params, f)


def load_data(scaler: str = "standard", split_val=True) -> tuple:
    """Loads teh original dataset, scales it and returns the train-val split.

    Args:
        scaler (str, optional): Name of the wanted scaler. Currently "standard" or "minmax". Defaults to "standard".
        split_val (bool, optional): Bool to determine if we split train in train/val.  Defaults to True.
    Returns:
        tuples: X_train, X_val, y_train, y_val, X_test
    """
    train = pd.read_csv(os.path.join(ORIG_DATA_PATH, "train.csv"))
    X_test = pd.read_csv(os.path.join(ORIG_DATA_PATH, "test.csv"))

    # Remove useless index column
    train = train.drop(columns=["Unnamed: 0"])
    X_test = X_test.drop(columns=["S.No"])

    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        columns_to_normalize = train.columns.drop("LABELS")
        # Fit transform train
        temp_x = train[columns_to_normalize].values
        temp_x = scaler.fit_transform(temp_x)
        df_temp = pd.DataFrame(temp_x, columns=columns_to_normalize, index=train.index)
        train[columns_to_normalize] = df_temp
        # Transform test
        temp_x = X_test[columns_to_normalize].values
        x_scaled = scaler.transform(temp_x)
        df_temp = pd.DataFrame(
            x_scaled, columns=columns_to_normalize, index=X_test.index
        )
        X_test[columns_to_normalize] = df_temp

    X, y = train.drop(columns=["LABELS"]), train["LABELS"]
    if split_val:
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=RANDOM_SEED)
        return X_train, X_val, y_train, y_val, X_test
    else:
        return X, y, X_test


def run_single_experiment(clf, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                          X_val: pd.DataFrame = None, y_val: pd.Series = None, use_comet: bool = False, do_save_preds=True):
    """Runs a single experiment with the provided classifier.

    Args:
        clf  : classifier. Assumed to have a fit/predict pattern.
        model_name (str): Name (type) of the model. ex: 'random_forest'.
        X_train (pd.Dataframe): Train set features
        y_train (pd.Series): Train set labels
        X_val (pd.Dataframe): val set features
        y_val (pd.Series): val set labels
        X_test (pd.Dataframe, optional): Test set features
        use_comet (bool, optional): Use comet.ml logging API. Defaults to False.
        do_save_preds (bool, optional): Bool to decide whehter the predictions on X_test are saved as a csv in data/preds/... Defaults to True.
    """
    with_val = X_val and y_val
    if use_comet and not with_val:
        print('Validation set is necessary to log metrics to Comet')
        return

    if use_comet and with_val:
        exp = Experiment(
            project_name="data-challenge-2",
            workspace="ift6390-datachallenge-2",
            auto_output_logging="default",
            api_key=os.environ.get("COMET_API"),
        )

    params = clf.get_params()
    clf.fit(X_train, y_train)

    if with_val:
        y_pred = clf.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        print(f"Accuracy: {accuracy:.4f} \t F1: {f1:.4f}")

        # these will be logged to your sklearn-demos project on Comet.ml
        params = {
            "random_state": RANDOM_SEED,
            "model_type": model_name,
            "scaler": "standard scaler",
            "param_grid": str(params),
            "stratify": True,
        }

        metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
        if use_comet:
            exp.log_dataset_hash(X_train)
            exp.log_parameters(params)
            exp.log_metrics(metrics)
            exp.add_tag(model_name)

    test_preds = clf.predict(X_test)
    if do_save_preds:
        save_preds(test_preds, model_name, params, metrics)
