import logging
# from pdb import set_trace

import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.linear_model import LogisticRegressionCV


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
             'missing': df[col].isnull().sum()*1.0/rows,
             'type': df[col].dtype}
        metafeatures.append(d)
    return pd.DataFrame(metafeatures)


def impute_fe(df, columns):
    """Impute columns with missing values"""
    logging.info('Imputing %s columns', len(columns))
    if not columns:
        return df
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    df.loc[:, columns] = imputer.fit_transform(df.loc[:, columns])
    return df


def le_columns(df, columns):
    """Feature engineering and cleaning, prepare for training.

    Args:
    -----
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


def train_classifier(train, cols_to_drop):
    X = train.drop(['id', 'poor', 'country'] + cols_to_drop, axis=1).as_matrix()
    y = train['poor'].as_matrix()
    logging.info('Training classifier')
    clf = LogisticRegressionCV(n_jobs=-1, scoring='neg_log_loss')
    clf.fit(X, y)
    return clf


def make_test(csv, cat_cols, cols_to_drop):
    test = pd.read_csv(csv)
    meta = get_metafeatures(test)
    cols_to_impute = meta.loc[(meta['missing'] > 0.0) & (meta['missing'] < 0.5), 'column']\
                         .tolist()
    test = impute_fe(test, cols_to_impute)
    test = le_columns(test, cat_cols)
    X = test.drop(['id', 'country'] + cols_to_drop, axis=1).as_matrix()
    return test, X


def make_sub(preds, test_feat, country):
    country_sub = pd.DataFrame(data=preds[:, 1],
                               columns=['poor'],
                               index=test_feat['id'])
    # add country code for joining later
    country_sub['country'] = country
    return country_sub[['country', 'poor']]


def main(country):
    """Pipeline for reading, building features and making submissions."""
    # for one country
    assert country in ['A', 'B', 'C']
    train = pd.read_csv('../data/raw/{}_hhold_train.csv'.format(country))
    meta = get_metafeatures(train)
    cols_to_impute = meta.loc[(meta['missing'] > 0.0) & (meta['missing'] < 0.5), 'column'].tolist()
    cols_to_drop = meta.loc[meta['missing'] > 0.5, 'column'].tolist()
    categorical_columns = meta.loc[meta['type'] == 'object', 'column'].tolist()
    train = impute_fe(train, cols_to_impute)
    train = le_columns(train, categorical_columns)
    clf = train_classifier(train, cols_to_drop)
    test, X = make_test('../data/raw/{}_hhold_test.csv'.format(country),
        categorical_columns, cols_to_drop)
    preds = clf.predict_proba(X)
    sub = make_sub(preds, test, country).reset_index()
    return sub


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level='DEBUG')
    sub_a = main('A')
    sub_b = main('B')
    sub_c = main('C')
    submissions = pd.concat([sub_a, sub_b, sub_c])
    submissions.to_csv('../data/processed/first_submission.csv', index=False)