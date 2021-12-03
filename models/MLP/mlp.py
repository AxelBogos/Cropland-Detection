import os
from pprint import pprint

import numpy as np
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from models.utils import run_single_experiment, load_orig_data, load_processed_data


def run_experiment(params=None, do_save_preds=False, use_comet=False, use_processed=False):
    name = os.path.basename(__file__).split('.')[0]
    if use_processed:
        X_train, X_val, y_train, y_val, X_test = load_processed_data(version='correlated_feature_dropped', split_val=True)
    else:
        X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
    if params is None:
        clf = MLPClassifier()
    else:
        clf = MLPClassifier(**params)
    run_single_experiment(clf=clf, model_name=name, X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val, X_test=X_test, use_comet=use_comet, do_save_preds=do_save_preds)


def run_optimization():
    def objective(trial, X, y):

        param_grid = {
            'alpha': trial.suggest_uniform('alpha',0.0001, 0.001),
            'learning_rate_init' : trial.suggest_loguniform('learning_rate_init',1e-5, 0.3),
            "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive']),
            "momentum": trial.suggest_float("momentum", 0, 0.99),
        }
        n_layers = trial.suggest_int('n_layers', 1, 5)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 1, 150))
        param_grid['max_iter'] = 500


        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = MLPClassifier(**param_grid,hidden_layer_sizes=tuple(layers))
            model.fit(X_train,y_train)
            preds = model.predict_proba(X_test)
            cv_scores[idx] = log_loss(y_test, preds)
        return np.mean(cv_scores)

    X, y, X_test = load_orig_data(scaler='standard', split_val=False)

    study = optuna.create_study(direction="minimize", study_name="MLP Classifier")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=60)
    print(f"\tBest params:")
    pprint(study.best_params)
    params = study.best_params

    name = os.path.basename(__file__).split('.')[0]
    clf = MLPClassifier(**params)
    X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
    run_single_experiment(clf=clf, model_name=name, X_train=X, y_train=y, X_val=X_val, y_val=y_val,
                          X_test=X_test, use_comet=False, do_save_preds=True)


def main():
    params = {'alpha': 0.0008850441826891523,
    'learning_rate': 'adaptive',
    'learning_rate_init': 1.7016199573556555e-05,
    'momentum': 0.5428082178458562,
    'hidden_layer_sizes': (132)}
    run_experiment(params, True, True, False)


if __name__ == "__main__":
    main()
