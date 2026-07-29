import os
import numpy as np
import pandas as pd

RAW = os.path.expanduser('~/Documents/2026MLpredictor/data/raw')
PROCESSED = os.path.expanduser('~/Documents/2026MLpredictor/data/processed')

ATLARGE_STATES = {'AK', 'DE', 'MT', 'ND', 'SD', 'VT', 'WY'}
MIDTERM_YEARS = [2010, 2014, 2018, 2022]


def load_raw_results():
    raw = pd.read_csv(os.path.join(RAW, 'mit_house_results.tab'), low_memory=False)
    df = raw[(raw['stage'] == 'GEN') & (raw['special'] == False) & (raw['writein'] == False)].copy()
    print('loaded raw results:', df.shape, flush=True)
    return df


def aggregate_to_district_year(df):
    df['is_dem_line'] = df['party'].isin(['DEMOCRAT', 'DEMOCRATIC-FARMER-LABOR'])
    df['is_rep_line'] = df['party'].isin(['REPUBLICAN'])

    candidate_party = df.groupby(['year', 'state_po', 'district', 'candidate']).agg(
        total_votes=('candidatevotes', 'sum'),
        on_dem_line=('is_dem_line', 'any'),
        on_rep_line=('is_rep_line', 'any'),
    ).reset_index()

    candidate_party['party'] = np.where(
        candidate_party['on_dem_line'], 'D',
        np.where(candidate_party['on_rep_line'], 'R', 'Other')
    )

    district_year = (
        candidate_party.groupby(['year', 'state_po', 'district', 'party'])['total_votes']
        .sum().unstack(fill_value=0).reset_index()
    )
    district_year.columns.name = None
    district_year = district_year.rename(columns={'D': 'dem_votes', 'R': 'rep_votes', 'Other': 'other_votes'})

    district_year['dem_share'] = np.where(
        (district_year['dem_votes'] > 0) & (district_year['rep_votes'] > 0),
        district_year['dem_votes'] / (district_year['dem_votes'] + district_year['rep_votes']),
        np.nan
    )
    district_year['uncontested'] = np.where(
        (district_year['dem_votes'] == 0) | (district_year['rep_votes'] == 0), True, False
    )

    house = district_year[(district_year['year'] >= 2010) & (district_year['year'] <= 2022)].copy()
    print('district-year rows:', house.shape, flush=True)
    return house


def get_lean_for_year(lean, year):
    if year <= 2018:
        col = 'lean_2018'
    elif year == 2020:
        col = 'lean_2020'
    else:
        col = 'lean_2022'
    return lean[['state_po', 'district', col]].rename(columns={col: 'partisan_lean'}).assign(year=year)


def merge_partisan_lean(house):
    lean = pd.read_csv(os.path.join(RAW, 'partisan_lean.csv'), low_memory=False)
    lean_long = pd.concat([get_lean_for_year(lean, y) for y in [2010, 2012, 2014, 2016, 2018, 2020, 2022]])

    house['district_key'] = np.where(
        (house['state_po'].isin(ATLARGE_STATES)) & (house['district'] == 0),
        1,
        house['district']
    )
    house = pd.merge(
        house, lean_long,
        left_on=['year', 'state_po', 'district_key'],
        right_on=['year', 'state_po', 'district'],
        how='left'
    )
    house = house.drop(columns=['district_y', 'district_key']).rename(columns={'district_x': 'district'})
    print('missing partisan_lean after merge:', house['partisan_lean'].isnull().sum(), flush=True)
    return house


def merge_econ_and_flags(house):
    econ = pd.read_csv(os.path.join(RAW, 'fred_econ.csv'), low_memory=False)
    house = pd.merge(house, econ, on='year', how='left')
    house['midterm_year'] = np.where(house['year'].isin(MIDTERM_YEARS), 1, 0)
    house['redistricted'] = np.where(house['year'] == 2022, 1, 0)
    return house


def add_incumbency_features(house):
    house = house.sort_values(['state_po', 'district', 'year'])
    house['dem_share_prev'] = house.groupby(['state_po', 'district'])['dem_share'].shift(1)
    house['incumbent_party'] = np.where(
        house['dem_share_prev'] > 0.5, 1,
        np.where(house['dem_share_prev'].isna(), 0, -1)
    )
    return house


def main():
    df = load_raw_results()
    house = aggregate_to_district_year(df)
    house = merge_partisan_lean(house)
    house = merge_econ_and_flags(house)
    house = add_incumbency_features(house)

    out_path = os.path.join(PROCESSED, 'house_week1_merged.csv')
    house.to_csv(out_path, index=False)
    print(f'saved {len(house)} rows to {out_path}', flush=True)


if __name__ == '__main__':
    main()
