import os
from models.utils import run_single_experiment, load_data
import lightgbm as lgbm

name = os.path.basename(__file__).split('.')[0]
X_train, X_val, y_train, y_val, X_test = load_data(scaler='standard', split_val=True)
clf = lgbm.LGBMClassifier()
run_single_experiment(clf=clf, model_name=name, X_train=X_train, y_train=y_train,
                      X_val=X_val, y_val=y_val, X_test=X_test, use_comet=False)
