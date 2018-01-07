"""Module to clean columns and build features for training model."""
import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_metafeatures(df):
    """
    Get metadata of the columns in the training set.

    Returns a metadata DataFrame.
    """
    logging.info('Creating metafeatures dataframe.')
    metafeatures = []
    rows = df.shape[0]
    for col in df.columns:
        d = {'column': col,
             'n_unique': df[col].nunique(),
             'missing': df[col].isnull().sum() * 1.0 / rows,
             'type': df[col].dtype}
        metafeatures.append(d)
    return pd.DataFrame(metafeatures)


def le_columns(df, columns):
    """Feature engineering and cleaning, prepare for training.

    Parameters:
    -----------
    df: dataframe
    columns: list of categorical columns

    Returns:
    --------
    Transformed dataframe
    """
    logging.info('LabelEncoder for %s columns', len(columns))
    le = LabelEncoder()
    for col in columns:
        df.loc[:, col] = le.fit_transform(df[col])
    return df


def get_cols(df):
    """Returns list of categorical columns and columns to drop."""
    meta = get_metafeatures(df)
    categorical_columns = meta.loc[meta['type'] == 'object', 'column'].tolist()
    cols_to_drop = meta.loc[meta['missing'] > 0.5, 'column'].tolist()
    return categorical_columns, cols_to_drop
