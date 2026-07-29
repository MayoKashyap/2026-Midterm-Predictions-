import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RAW = os.path.expanduser('~/Documents/2026MLpredictor/data/raw')
MODELS = os.path.expanduser('~/Documents/2026MLpredictor/models')

CCES_COLUMNS = ['birthyr', 'educ', 'pid7', 'newsint', 'inputstate', 'CC18_401', 'CL_matched', 'CL_2018gvm']

EDUC_MAP = {
    'No HS': 1, 'High school graduate': 2, 'Some college': 3,
    '2-year': 4, '4-year': 5, 'Post-grad': 6
}
PID7_MAP = {
    'Strong Democrat': 1, 'Not very strong Democrat': 2, 'Lean Democrat': 3,
    'Independent': 4, 'Lean Republican': 5, 'Not very strong Republican': 6,
    'Strong Republican': 7
}
NEWSINT_MAP = {
    'Hardly at all': 1, 'Only now and then': 2, 'Some of the time': 3, 'Most of the time': 4
}


def load_verified_respondents():
    cces18 = pd.read_csv(
        os.path.join(RAW, 'CCES18_Common_OUTPUT_vv_topost.csv'),
        usecols=CCES_COLUMNS,
        low_memory=False
    )
    print('raw respondents:', len(cces18), flush=True)

    cces18 = cces18[cces18['CL_matched'] == 'Yes'].copy()
    cces18['voted'] = cces18['CL_2018gvm'].notna().astype(int)
    print('verified respondents:', len(cces18), flush=True)
    print(cces18['voted'].value_counts(normalize=True), flush=True)
    return cces18


def build_features(cces18):
    cces18['age'] = 2018 - cces18['birthyr']
    cces18['educ_ord'] = cces18['educ'].map(EDUC_MAP)
    cces18['pid7_ord'] = cces18['pid7'].map(PID7_MAP)
    cces18['newsint_ord'] = cces18['newsint'].map(NEWSINT_MAP)

    state_dummies = pd.get_dummies(cces18['inputstate'], prefix='state')
    cc401_dummies = pd.get_dummies(cces18['CC18_401'], prefix='cc401', dummy_na=True)

    features = pd.concat([
        cces18[['age', 'educ_ord', 'pid7_ord', 'newsint_ord']],
        state_dummies,
        cc401_dummies
    ], axis=1)
    target = cces18['voted']

    complete_rows = ~features[['age', 'educ_ord', 'pid7_ord', 'newsint_ord']].isna().any(axis=1)
    features = features.loc[complete_rows]
    target = target.loc[features.index]

    print('final training rows:', len(features), flush=True)
    print('final voted rate:', target.mean(), flush=True)
    return features, target


def train_and_evaluate(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print('train accuracy:', model.score(X_train, y_train), flush=True)
    print('test accuracy:', model.score(X_test, y_test), flush=True)

    coefs = pd.Series(model.coef_[0], index=features.columns).sort_values()
    print(coefs.loc[['age', 'educ_ord', 'pid7_ord', 'newsint_ord']], flush=True)

    return model


def main():
    cces18 = load_verified_respondents()
    features, target = build_features(cces18)
    model = train_and_evaluate(features, target)

    os.makedirs(MODELS, exist_ok=True)
    joblib.dump({'model': model, 'columns': list(features.columns)}, os.path.join(MODELS, 'likely_voter_model.joblib'))
    print('saved model to models/likely_voter_model.joblib', flush=True)


if __name__ == '__main__':
    main()
