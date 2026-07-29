import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import norm

RAW = os.path.expanduser('~/Documents/2026MLpredictor/data/raw')
PROCESSED = os.path.expanduser('~/Documents/2026MLpredictor/data/processed')
MODELS = os.path.expanduser('~/Documents/2026MLpredictor/models')

FEATURES = [
    'partisan_lean', 'gdp_growth_q2', 'unemp_oct', 'log_gdp_growth',
    'midterm_year', 'redistricted', 'incumbent_party', 'dem_share_prev',
    'pres_approval'
]

BACKTEST_RMSE = 0.052


def assemble_2026_features():
    pvi = pd.read_csv(os.path.join(RAW, 'pvi_119th.csv'))
    share_prev = pd.read_csv(os.path.join(RAW, 'house_results_2024.csv'))
    incumbency = pd.read_csv(os.path.join(RAW, 'incumbency_2026.csv'))
    econ = pd.read_csv(os.path.join(RAW, 'econ_2026.csv'))

    feat = pvi.merge(share_prev, on=['state_po', 'district'], how='left')
    feat = feat.merge(incumbency, on=['state_po', 'district'], how='left')

    feat['gdp_growth_q2'] = econ['gdp_growth_q2'].iloc[0]
    feat['unemp_oct'] = econ['unemp_oct'].iloc[0]
    feat['log_gdp_growth'] = econ['log_gdp_growth'].iloc[0]
    feat['pres_approval'] = econ['pres_approval'].iloc[0]

    feat['midterm_year'] = 1
    feat['redistricted'] = 0

    print('2026 feature matrix shape:', feat.shape, flush=True)
    print('missing values per feature:', flush=True)
    print(feat[FEATURES].isna().sum(), flush=True)
    return feat


def rating(win_prob):
    if win_prob >= 0.85:
        return 'Safe D'
    if win_prob >= 0.65:
        return 'Lean D'
    if win_prob >= 0.35:
        return 'Toss-up'
    if win_prob >= 0.15:
        return 'Lean R'
    return 'Safe R'


def main():
    feat = assemble_2026_features()

    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODELS, 'house_fundamentals.json'))

    preds = model.predict(feat[FEATURES])
    win_prob = norm.cdf((preds - 0.5) / BACKTEST_RMSE)
    print(f'projected Dem seats: {(win_prob > 0.5).sum()}', flush=True)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(feat[FEATURES])

    hist = pd.read_csv(os.path.join(PROCESSED, 'house_week1_merged.csv'))
    actual_2022 = hist[hist['year'] == 2022][['state_po', 'district', 'dem_share']].rename(
        columns={'dem_share': 'dem_share_2022'}
    )
    feat = feat.merge(actual_2022, on=['state_po', 'district'], how='left')
    feat['pred_dem_share'] = preds
    feat['win_prob_dem'] = win_prob

    records = []
    for i, row in feat.iterrows():
        sv = shap_vals[i]
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]
        shap_top3 = [{'feature': FEATURES[j], 'impact': round(float(sv[j]), 4)} for j in top3_idx]
        records.append({
            'state_po': row['state_po'],
            'district': int(row['district']),
            'pred_dem_share': round(float(row['pred_dem_share']), 4),
            'win_prob_dem': round(float(row['win_prob_dem']), 4),
            'rating': rating(row['win_prob_dem']),
            'dem_share_2024': round(float(row['dem_share_prev']), 4) if pd.notna(row.get('dem_share_prev')) else None,
            'dem_share_2022': round(float(row['dem_share_2022']), 4) if pd.notna(row.get('dem_share_2022')) else None,
            'shap': shap_top3,
        })

    out_path = os.path.join(PROCESSED, 'predictions_2026.json')
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'exported {len(records)} districts to predictions_2026.json', flush=True)


if __name__ == '__main__':
    main()
