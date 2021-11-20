from sklearn.ensemble import RandomForestClassifier
import os
from models.utils import run_single_experiment

clf = RandomForestClassifier()
run_single_experiment(clf=clf, model_name=os.path.basename(__file__).split('.')[0], scaler='standard')
