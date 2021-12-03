import os
from pprint import pprint

import numpy as np
import optuna
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from models.utils import run_single_experiment, load_orig_data, load_processed_data


def run_experiment(params=None, do_save_preds=False, use_comet=False, use_processed=False):
	name = os.path.basename(__file__).split('.')[0]
	if use_processed:
		X_train, X_val, y_train, y_val, X_test = load_processed_data(version='correlated_feature_dropped',
		                                                             split_val=True)
	else:
		X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
	if params is None:
		clf = xgb.XGBClassifier(objective="binary:logistic", verbose=-1)
	else:
		clf = xgb.XGBClassifier(**params)
	run_single_experiment(clf=clf, model_name=name, X_train=X_train, y_train=y_train,
	                      X_val=X_val, y_val=y_val, X_test=X_test, use_comet=use_comet, do_save_preds=do_save_preds)


def run_optimization():
	def objective(trial, X, y):

		param_grid = {
			"silent": 1,
			"objective": "binary:logistic",
			"eval_metric": "logloss",
			"booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
			"scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 10),
			"lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
			"alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
			"n_estimators": trial.suggest_int("n_estimators", 100, 4000)
		}

		if param_grid["booster"] == "gbtree" or param_grid["booster"] == "dart":
			param_grid["max_depth"] = trial.suggest_int("max_depth", 1, 12)
			param_grid["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
			param_grid["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
			param_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
		if param_grid["booster"] == "dart":
			param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
			param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
			param_grid["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
			param_grid["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

		cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
		cv_scores = np.empty(5)
		for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
			train = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx])
			val = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx])

			# Add a callback for pruning.
			clf = xgb.train(param_grid, train, evals=[(val, "validation")],
			                callbacks=[XGBoostPruningCallback(trial, "validation-logloss")])
			y_preds = clf.predict(val)
			score = log_loss(y[val_idx], y_preds)
			cv_scores[idx] = score
		return np.mean(cv_scores)

	X, y, X_test = load_orig_data(scaler='standard', split_val=False)
	study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
	                            direction="minimize", study_name="XGB Classifier")
	func = lambda trial: objective(trial, X, y)
	study.optimize(func, n_trials=75)
	print(f"\tBest params:")
	pprint(study.best_params)
	params = study.best_params

	name = os.path.basename(__file__).split('.')[0]
	clf = xgb.XGBClassifier(**params, verbose=-1)
	X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
	run_single_experiment(clf=clf, model_name=name, X_train=X, y_train=y, X_val=X_val, y_val=y_val,
	                      X_test=X_test, use_comet=True, do_save_preds=True)


def main():
	best_params = {'objective': 'binary:logistic', 'use_label_encoder': True,
               'base_score': None, 'booster': 'gbtree', 'colsample_bylevel':
               None, 'colsample_bynode': None, 'colsample_bytree': None,
               'gamma': 1.565090195551529e-05, 'gpu_id': None,
               'importance_type': 'gain', 'interaction_constraints': None,
               'learning_rate': None, 'max_delta_step': None, 'max_depth':
               12, 'min_child_weight': None, 'monotone_constraints': None, 'n_estimators': 3882, 'n_jobs':
               None, 'num_parallel_tree': None, 'random_state': None,
               'reg_alpha': None, 'reg_lambda': None, 'scale_pos_weight': 1,
               'subsample': None, 'tree_method': None, 'validate_parameters':
               None, 'verbosity': None, 'lambda': 0.8057909191717765,
               'alpha': 2.2192333922111406e-06, 'eta': 0.2765417444751599,
               'grow_policy': 'depthwise', 'verbose': -1}
	run_experiment(best_params,do_save_preds=True,use_comet=True,use_processed=False)


if __name__ == "__main__":
	main()
