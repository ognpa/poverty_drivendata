"""Module to generate final predictions and make submissions."""
import os
import logging

import pandas as pd

from src.models.train_model import prepare_data, cv_setup
from src import DATA_DIR


def get_predictions(country, pipe, params):
    """Get predictions using the best model from grid search or hyperopt search."""
    hhold_csv_test = os.path.join(DATA_DIR, 'raw', '{}_hhold_test.csv'.format(country))
    test = pd.read_csv(hhold_csv_test)
    X_test, _ = prepare_data(test)
    logging.debug('Fitting test data with params %s', params)
    preds = pipe.set_params(**params).predict_proba(X_test)
    return preds, test


def submit_predictions(preds, test, country):
    """Make submission."""
    country_sub = pd.DataFrame(data=preds[:, 1],
                               columns=['poor'],
                               index=test['id'])
    # add country code for joining later
    country_sub['country'] = country
    # write submission
    return country_sub[['country', 'poor']].reset_index()


def main(country):
    """Everything packaged here. Trains first, then predicts."""
    pipe, params, score = cv_setup(country)
    logging.info('Now, computing predictions')
    preds, test = get_predictions(country, pipe, params)
    return submit_predictions(preds, test, country)
