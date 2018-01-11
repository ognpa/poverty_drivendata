"""Main file for testing the results and generating submissions."""
import logging
import os
import argparse

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.models import train_model
from src import DATA_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict poverty')
    parser.add_argument('country', help='A or B or C')
    args = parser.parse_args()
    default_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=default_format)
    cachedir = '/tmp/xgb_training_{}/'.format(args.country)
    estimators = [('clf', XGBClassifier())]
    param_grid = [{
        # 'imputer__strategy': ['mean', 'median'],
        'clf__objective': ['binary:logistic']
    }]
    pipe = Pipeline(estimators, memory=cachedir)
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_log_loss', cv=10,
                        verbose=2, n_jobs=-1)
    submission = train_model.main(args.country, grid)
    submission.to_csv(os.path.join(DATA_DIR, 'interim', 'xgb_sub_{}.csv'.format(args.country)),
                      index=False)
