import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

PROCESSED = os.path.expanduser('~/Documents/2026MLpredictor/data/processed')
RAW = os.path.expanduser('~/Documents/2026MLpredictor/data/raw')
MODELS = os.path.expanduser('~/Documents/2026MLpredictor/models')

FEATURES = [
    'partisan_lean', 'gdp_growth_q2', 'unemp_oct', 'log_gdp_growth',
    'midterm_year', 'redistricted', 'incumbent_party', 'dem_share_prev',
    'pres_approval'
]

AT_LARGE = ['AK', 'DE', 'MT', 'ND', 'SD', 'VT', 'WY']

FTE_FILES = {
    2018: os.path.join(RAW, 'fte_2018_house_forecast.csv'),
    2020: os.path.join(RAW, 'fte_2020_house_forecast.csv'),
    2022: os.path.join(RAW, 'fte_2022_house_forecast.csv'),
}


def load_dataset():
    df = pd.read_csv(os.path.join(PROCESSED, 'house_week1_merged.csv'), low_memory=False)
    df = df[df['uncontested'] == False]

    approval = pd.read_csv(os.path.join(RAW, 'pres_approval.csv'))
    df = df.merge(approval[['year', 'pres_approval']], on='year', how='left')
    print(f'dataset shape: {df.shape}, NaNs in pres_approval: {df["pres_approval"].isna().sum()}', flush=True)
    return df


def fit_model(train_df):
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(train_df[FEATURES], train_df['dem_share'])
    return model


def score(model, test_df):
    preds = model.predict(test_df[FEATURES])
    rmse = np.sqrt(mean_squared_error(test_df['dem_share'], preds))
    win_prob = norm.cdf((preds - 0.5) / rmse)
    actual = (test_df['dem_share'].values > 0.5).astype(float)
    brier = np.mean((win_prob - actual) ** 2)
    return preds, rmse, brier


def benchmark_against_fte(test_df, test_year, our_rmse, our_brier):
    test_df = test_df.copy()
    test_df['district_fixed'] = test_df.apply(
        lambda row: 1 if row['state_po'] in AT_LARGE else row['district'], axis=1
    )
    test_df['match_key'] = test_df['state_po'] + '-' + test_df['district_fixed'].astype(str)

    fte = pd.read_csv(FTE_FILES[test_year])
    merged = test_df.merge(fte, left_on='match_key', right_on='district', how='inner')

    fte_vs = merged['fte_voteshare_dem'].astype(float)
    if fte_vs.max() > 1:
        fte_vs = fte_vs / 100

    actual = (merged['dem_share'].values > 0.5).astype(float)
    fte_rmse = np.sqrt(mean_squared_error(merged['dem_share'], fte_vs))
    fte_brier = np.mean((merged['fte_winner_dem_prob'].astype(float) - actual) ** 2)

    print(
        f'{test_year}  Our RMSE={our_rmse:.4f} Brier={our_brier:.4f}  |  '
        f'538 RMSE={fte_rmse:.4f} Brier={fte_brier:.4f}  (n={len(merged)})',
        flush=True
    )
    return {'year': test_year, 'n': len(merged), 'our_rmse': our_rmse, 'fte_rmse': fte_rmse,
            'our_brier': our_brier, 'fte_brier': fte_brier}


def run_multi_cycle_backtest(df):
    comparison = []
    for test_year, train_cutoff in [(2018, 2016), (2020, 2018), (2022, 2020)]:
        train_df = df[df['year'] <= train_cutoff]
        test_df = df[df['year'] == test_year]

        model = fit_model(train_df)
        _, rmse, brier = score(model, test_df)
        comparison.append(benchmark_against_fte(test_df, test_year, rmse, brier))

    print(flush=True)
    print(pd.DataFrame(comparison)[['year', 'our_rmse', 'fte_rmse', 'our_brier', 'fte_brier']].to_string(index=False), flush=True)
    return comparison


def save_final_model(df):
    final_model = fit_model(df)
    os.makedirs(MODELS, exist_ok=True)
    final_model.save_model(os.path.join(MODELS, 'house_fundamentals.json'))
    print('saved final model to models/house_fundamentals.json', flush=True)
    return final_model


def main():
    df = load_dataset()
    run_multi_cycle_backtest(df)
    save_final_model(df)


if __name__ == '__main__':
    main()
