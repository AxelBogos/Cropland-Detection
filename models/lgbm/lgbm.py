from comet_ml import Experiment
import os
from pprint import pprint

import lightgbm as lgbm
import numpy as np
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from models.utils import run_single_experiment, load_orig_data, load_processed_data


def run_experiment(params=None, do_save_preds=False, use_comet=False, processed_version=None):
    name = os.path.basename(__file__).split('.')[0]
    if processed_version is not None:
        X_train, X_val, y_train, y_val, X_test = load_processed_data(version=processed_version, split_val=True)
    else:
        X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
    if params is None:
        clf = lgbm.LGBMClassifier(objective="binary", verbose=-1)
    else:
        clf = lgbm.LGBMClassifier(**params)
    run_single_experiment(clf=clf, model_name=name, X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val, X_test=X_test, use_comet=use_comet, do_save_preds=do_save_preds)


def run_optimization(save_plots=False, processed_version = None):
    def objective(trial, X, y):
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 8, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 5000, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1, step=0.1),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1, step=0.1),
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgbm.LGBMClassifier(objective="binary", **param_grid, verbose=-1)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="binary_logloss",
                early_stopping_rounds=75,
                callbacks=[LightGBMPruningCallback(trial, "binary_logloss")],
            )
            preds = model.predict_proba(X_test)
            cv_scores[idx] = log_loss(y_test, preds)
        return np.mean(cv_scores)

    if processed_version is None:
        X, y, X_test = load_orig_data(scaler='standard', split_val=False)
    else:
        X, y, X_test = load_processed_data(processed_version, split_val=False)

    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y)
    study.optimize(func, n_trials=150)

    if save_plots:
        # Save study plots
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html('optuna_optim_history.html')
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html('hyperparam_importance.html')
    print(f"\tBest params:")
    pprint(study.best_params)
    params = study.best_params
    return params
    # name = os.path.basename(__file__).split('.')[0]
    # clf = lgbm.LGBMClassifier(**params, verbose=-1)
    # X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
    # run_single_experiment(clf=clf, model_name=name, X_train=X, y_train=y, X_val=X_val, y_val=y_val,
    #                       X_test=X_test, use_comet=True, do_save_preds=True)


def main():
    best_params = {'boosting_type': 'gbdt', 'class_weight': None,
                   'colsample_bytree': 1.0, 'importance_type': 'split',
                   'learning_rate': 0.2290373641751674, 'max_depth': 11,
                   'min_child_samples': 20, 'min_child_weight': 0.001,
                   'min_split_gain': 0.0, 'n_estimators': 4600, 'n_jobs': -1,
                   'num_leaves': 728, 'objective': None, 'random_state': None,
                   'reg_alpha': 0.0, 'reg_lambda': 0.0, 'silent': True,
                   'subsample': 1.0, 'subsample_for_bin': 200000,
                   'subsample_freq': 0, 'min_data_in_leaf': 300, 'lambda_l1': 5,
                   'lambda_l2': 30, 'min_gain_to_split': 0.05652246284460086,
                   'bagging_fraction': 0.8, 'feature_fraction': 0.4, 'verbose': -1}

    # run_experiment(params=best_params, do_save_preds=True, use_comet=True, use_processed=False)
    agg_4params = run_optimization(save_plots=True)
    run_experiment(params=agg_4params, do_save_preds=True, use_comet=True, processed_version='aggregated_4')


if __name__ == "__main__":
    main()
