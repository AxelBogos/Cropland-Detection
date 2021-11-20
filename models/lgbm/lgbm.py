import os
from models.utils import run_single_experiment
import lightgbm as lgbm
clf = lgbm.LGBMClassifier()
run_single_experiment(clf=clf, model_name=os.path.basename(__file__).split('.')[0], scaler='standard')
