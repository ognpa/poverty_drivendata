"""Main file for testing the results and generating submissions."""
import logging, os
from tempfile import mkdtemp

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.models import train_model
from src import DATA_DIR

if __name__ == '__main__':
    default_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=default_format)
    cachedir = mkdtemp()
    estimators = [('imputer', Imputer()), ('clf', XGBClassifier())]
    param_grid = [{
        'imputer__strategy': ['mean', 'median'],
        'clf__objective': ['binary:logistic']
    }]
    pipe = Pipeline(estimators, memory=cachedir)
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_log_loss', cv=5,
                        verbose=1, n_jobs=-1)
    sub_a = train_model.main('A', grid)
    sub_a.to_csv(os.path.join(DATA_DIR, 'interim', 'xgb_sub_A.csv'), index=False)
