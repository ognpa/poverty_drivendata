import os
import logging

import pandas as pd

# preprocessing and feature selection
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit


from src.features.build_features import get_cols, le_columns
from src import DATA_DIR


def get_X_y(df):
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


def local_cv_score(country, grid):
    """Check local cv scores using a train-test split of 80-20."""
    hhold_csv_train = os.path.join(DATA_DIR, 'raw', '{}_hhold_train.csv'.format(country))
    df = pd.read_csv(hhold_csv_train)
    X, y = get_X_y(df)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    grid.fit(X_train, y_train)
    preds = grid.best_estimator_.predict_proba(X_test)[:, 1]
    return log_loss(y_test, preds)


def get_predictions(country, grid):
    """Get predictions using the best model from grid search"""
    hhold_csv_train = os.path.join(DATA_DIR, 'raw', '{}_hhold_train.csv'.format(country))
    df = pd.read_csv(hhold_csv_train)
    X, y = get_X_y(df)
    grid.fit(X, y)
    hhold_csv_test = os.path.join(DATA_DIR, 'raw', '{}_hhold_test.csv'.format(country))
    test = pd.read_csv(hhold_csv_test)
    X_test, _ = get_X_y(test)
    preds = grid.best_estimator_.predict_proba(X_test)
    return preds, test


def make_subs(preds, test_feat, country):
    """Make submission."""
    country_sub = pd.DataFrame(data=preds[:, 1],
                               columns=['poor'],
                               index=test_feat['id'])
    # add country code for joining later
    country_sub['country'] = country
    # write submission
    return country_sub[['country', 'poor']].reset_index()


def main(country, grid):
    """Everything packaged here."""
    preds, test = get_predictions(country, grid)
    return make_subs(preds, test, country)
