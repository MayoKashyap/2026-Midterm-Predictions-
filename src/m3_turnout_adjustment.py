import os

import joblib
import pandas as pd

MODELS = os.path.expanduser('~/Documents/2026MLpredictor/models')

NEWSINT_POPULATION_AVERAGE = 3.23


def load_model():
    saved = joblib.load(os.path.join(MODELS, 'likely_voter_model.joblib'))
    return saved['model'], saved['columns']


def build_feature_row(columns, age, educ_ord, pid7_ord):
    row = pd.Series(0.0, index=columns)
    row['age'] = age
    row['educ_ord'] = educ_ord
    row['pid7_ord'] = pid7_ord
    row['newsint_ord'] = NEWSINT_POPULATION_AVERAGE
    return row


def adjust_poll_topline(poll_groups):
    model, columns = load_model()

    weighted_support = 0.0
    weighted_turnout = 0.0

    for group in poll_groups:
        row = build_feature_row(columns, group['age'], group['educ_ord'], group['pid7_ord'])
        turnout_prob = model.predict_proba(pd.DataFrame([row]))[:, 1][0]

        effective_weight = group['share_of_poll'] * turnout_prob
        weighted_support += effective_weight * group['pct_support_dem']
        weighted_turnout += effective_weight

    return weighted_support / weighted_turnout


if __name__ == '__main__':
    example_poll = [
        {'age': 25, 'educ_ord': 4, 'pid7_ord': 2, 'share_of_poll': 0.25, 'pct_support_dem': 0.65},
        {'age': 45, 'educ_ord': 3, 'pid7_ord': 4, 'share_of_poll': 0.35, 'pct_support_dem': 0.50},
        {'age': 70, 'educ_ord': 3, 'pid7_ord': 6, 'share_of_poll': 0.40, 'pct_support_dem': 0.35},
    ]

    raw_topline = sum(g['share_of_poll'] * g['pct_support_dem'] for g in example_poll)
    adjusted_topline = adjust_poll_topline(example_poll)

    print(f'raw topline: {raw_topline:.3f}', flush=True)
    print(f'turnout-adjusted topline: {adjusted_topline:.3f}', flush=True)
