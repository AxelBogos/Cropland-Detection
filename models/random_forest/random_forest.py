from comet_ml import Experiment
import os
from sklearn.ensemble import RandomForestClassifier

from models.utils import run_single_experiment, load_orig_data, load_processed_data


def run_experiment(params=None, do_save_preds=False, use_comet=False, processed_version=None):
    name = os.path.basename(__file__).split('.')[0]
    if processed_version is not None:
        X_train, X_val, y_train, y_val, X_test = load_processed_data(version=processed_version, split_val=True)
    else:
        X_train, X_val, y_train, y_val, X_test = load_orig_data(scaler='standard', split_val=True)
    if params is None:
        clf = RandomForestClassifier()
    else:
        clf = RandomForestClassifier(**params)
    run_single_experiment(clf=clf, model_name=name, X_train=X_train, y_train=y_train,
                          X_val=X_val, y_val=y_val, X_test=X_test, use_comet=use_comet, do_save_preds=do_save_preds)
def main():
    run_experiment(params=None, do_save_preds=True, use_comet=True)

if __name__ == "__main__":
    main()