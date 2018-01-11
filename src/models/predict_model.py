"""Module to generate final predictions and make submissions."""
import os
import logging

from src.models.train_model import prepare_data
from src import DATA_DIR

def get_predictions(country, grid):
    """Get predictions using the best model from grid search"""
    hhold_csv_train = os.path.join(DATA_DIR, 'raw', '{}_hhold_train.csv'.format(country))
    df = pd.read_csv(hhold_csv_train)
    X, y = prepare_data(df)
    grid.fit(X, y)
    hhold_csv_test = os.path.join(DATA_DIR, 'raw', '{}_hhold_test.csv'.format(country))
    test = pd.read_csv(hhold_csv_test)
    X_test, _ = prepare_data(test)
    preds = grid.best_estimator_.predict_proba(X_test)
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


def main(country, grid):
    """Everything packaged here."""
    preds, test = get_predictions(country, grid)
    return submit_predictions(preds, test, country)