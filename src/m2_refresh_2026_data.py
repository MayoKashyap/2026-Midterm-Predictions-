import io
import os

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv(os.path.expanduser('~/Documents/2026MLpredictor/.env'))

RAW = os.path.expanduser('~/Documents/2026MLpredictor/data/raw')

TRUMP_APPROVAL = 41.0

VALID_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA',
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
    'VA', 'WA', 'WV', 'WI', 'WY'
]

STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}


def refresh_econ():
    fred = Fred(api_key=os.environ['FRED_API_KEY'])

    gdp = fred.get_series('A191RL1Q225SBEA')
    gdp_2026_q1 = gdp['2026-01-01':'2026-03-31'].iloc[-1]

    unemp = fred.get_series('UNRATE')
    unemp_latest = unemp['2026-01-01':].iloc[-1]

    log_gdp = np.log(gdp_2026_q1) if gdp_2026_q1 > 0 else -np.log(-gdp_2026_q1)

    econ_2026 = {
        'year': 2026,
        'gdp_growth_q2': gdp_2026_q1,
        'unemp_oct': unemp_latest,
        'log_gdp_growth': log_gdp,
        'pres_approval': TRUMP_APPROVAL,
    }
    pd.DataFrame([econ_2026]).to_csv(os.path.join(RAW, 'econ_2026.csv'), index=False)
    print(f'saved econ_2026.csv: gdp_growth_q2={gdp_2026_q1}, unemp_oct={unemp_latest}, log_gdp_growth={log_gdp:.4f}', flush=True)


def parse_district(d):
    parts = d.split()
    last = parts[-1]
    state_name = ' '.join(parts[:-1])
    district = 0 if last.lower() == 'at-large' else int(last)
    return STATE_ABBREV[state_name], district


def parse_pvi(pvi):
    if pvi == 'EVEN':
        return 0.0
    direction = 1 if pvi.startswith('D') else -1
    return direction * float(pvi.split('+')[1])


def refresh_cook_pvi():
    r = requests.get(
        'https://en.wikipedia.org/wiki/Cook_Partisan_Voting_Index',
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    pvi_tables = pd.read_html(io.StringIO(r.text))
    pvi_raw = pvi_tables[0][['District', 'PVI']].copy()

    pvi_raw[['state_po', 'district']] = pvi_raw['District'].apply(lambda x: pd.Series(parse_district(x)))
    pvi_raw['partisan_lean'] = pvi_raw['PVI'].apply(parse_pvi)
    pvi_clean = pvi_raw[['state_po', 'district', 'partisan_lean']].copy()

    pvi_clean.to_csv(os.path.join(RAW, 'pvi_119th.csv'), index=False)
    print(f'saved pvi_119th.csv: {len(pvi_clean)} districts', flush=True)


def refresh_2024_results_and_incumbency():
    url = 'https://michaelminn.net/tutorials/data/2024-electoral-districts.csv'
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    results_2024 = pd.read_csv(io.StringIO(r.text))

    results_2024['state_po'] = results_2024['Name'].str.split('-').str[0]
    results_2024['district_raw'] = results_2024['Name'].str.split('-').str[1]
    results_2024['district'] = results_2024['district_raw'].apply(lambda x: 0 if not str(x).isdigit() else int(x))
    results_2024 = results_2024[results_2024['state_po'].isin(VALID_STATES)].copy()

    dem = results_2024['Votes_Dem_2024'].fillna(0)
    rep = results_2024['Votes_GOP_2024'].fillna(0)
    total = dem + rep
    results_2024['dem_share_prev'] = (dem / total).fillna(0)
    results_2024['winner_party'] = (dem > rep).map({True: 1, False: -1})

    share_2024 = results_2024[['state_po', 'district', 'dem_share_prev']].copy()
    share_2024.to_csv(os.path.join(RAW, 'house_results_2024.csv'), index=False)
    print(f'saved house_results_2024.csv: {len(share_2024)} districts', flush=True)

    incumbency_2026 = results_2024[['state_po', 'district', 'winner_party']].rename(
        columns={'winner_party': 'incumbent_party'}
    )
    incumbency_2026.to_csv(os.path.join(RAW, 'incumbency_2026.csv'), index=False)
    print(f'saved incumbency_2026.csv: {len(incumbency_2026)} districts', flush=True)


def main():
    refresh_econ()
    refresh_cook_pvi()
    refresh_2024_results_and_incumbency()


if __name__ == '__main__':
    main()
