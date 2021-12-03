import os

import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from models.utils import run_single_experiment, load_orig_data, load_processed_data


def run_experiment(models=None, do_save_preds=False, use_comet=False, use_processed=False):
    name = os.path.basename(__file__).split('.')[0]
    if use_processed:
        X_train, X_val, y_train, y_val, X_test = load_processed_data(version='correlated_feature_dropped', split_val=True)
    else:
        X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)

    clf = StackingClassifier(estimators=models,final_estimator=LogisticRegression(), cv=5)
    run_single_experiment(clf=clf, model_name=name, X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val, X_test=X_test, use_comet=use_comet, do_save_preds=do_save_preds)



def main():
    lgbm_best_params = {'boosting_type': 'gbdt', 'class_weight': None,
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
    xgb_best_params = {'objective': 'binary:logistic', 'use_label_encoder': True,
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
    mlp_best_params = {'alpha': 0.0008850441826891523,
        'learning_rate': 'adaptive',
        'learning_rate_init': 1.7016199573556555e-05,
        'momentum': 0.5428082178458562,
        'hidden_layer_sizes': (132)}
    models = list()
    models.append(('LGBM', lgbm.LGBMClassifier(**lgbm_best_params)))
    models.append(('XGB', xgb.XGBClassifier(**xgb_best_params)))
    models.append(('MLP', MLPClassifier()))

    run_experiment(models=models, do_save_preds=True, use_comet=True, use_processed=False)

if __name__ == "__main__":
    main()
