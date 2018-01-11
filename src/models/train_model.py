"""
Module to fit estimators and perform local cross-validation.
"""
import os
from pdb import set_trace

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

from src import DATA_DIR
from src.features.build_features import get_cols, le_columns


def prepare_data(df):
    """Drop columns, build features, impute missing values and return X and y,
    for training."""
    cat_cols, cols_to_drop = get_cols(df)
    df = df.pipe(le_columns, cat_cols)
    feature_columns = set(df.columns) - set(['id', 'country', 'poor']) - set(cols_to_drop)
    X = df.loc[:, feature_columns].as_matrix()
    try:
        y = df.loc[:, 'poor'].as_matrix()
    except KeyError:
        y = None
    return X, y


def estimate_local_cv(X, y):
    pipe = Pipeline([('imputer', Imputer()),
                     ('clf', LogisticRegression(random_state=0))])
    gs = GridSearchCV(
        estimator=pipe,
        scoring='neg_log_loss',
        param_grid=[{
            'clf__C': [0.01, 0.1, 0.5, 1]
        }],
        cv=5,
        n_jobs=-1,
        verbose=3
    )
    set_trace()
    scores = cross_val_score(gs, X, y, scoring='neg_log_loss', cv=2)
    return scores


def cv_setup(country):
    """Helper function to return the cv scores for a country. Overall log_loss is mean
    of log_loss scores of all countries."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'raw', '{}_hhold_train.csv'.format(country)))
    X, y = prepare_data(df)
    scores = estimate_local_cv(X, y)
    return np.mean(scores)


if __name__ == '__main__':
    print(cv_setup('C'))
