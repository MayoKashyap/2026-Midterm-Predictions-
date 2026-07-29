import json
import os
import time

import numpy as np
import voyageai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(os.path.expanduser('~/Documents/2026MLpredictor/.env'))

DOCS = os.path.expanduser('~/Documents/2026MLpredictor/docs')
PROCESSED = os.path.expanduser('~/Documents/2026MLpredictor/data/processed')

STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def build_methodology_chunks():
    with open(os.path.join(DOCS, 'about.html')) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    chunks = []
    for section in soup.select('details.accordion'):
        for svg in section.select('svg'):
            svg.decompose()
        title = section.select_one('summary').get_text(strip=True)
        body = section.select_one('.accordion-body-inner')
        text = body.get_text(separator=' ', strip=True) if body else ''
        if len(text) > 50:
            chunks.append({'source': 'methodology', 'title': title, 'text': f'{title}: {text}'})

    print('methodology chunks:', len(chunks), flush=True)
    return chunks


def district_to_paragraph(row):
    state = STATE_NAMES.get(row['state_po'], row['state_po'])
    dist_label = f"{state}'s {ordinal(row['district'])} congressional district"

    pred_pct = row['pred_dem_share'] * 100
    win_pct = row['win_prob_dem'] * 100
    top_factors = ', '.join(f"{s['feature']} ({s['impact']:+.3f})" for s in row['shap'][:3])

    sentence = (
        f"{dist_label} is rated {row['rating']}. The model predicts a {pred_pct:.1f}% "
        f"Democratic vote share, giving Democrats a {win_pct:.1f}% win probability. "
        f"The top SHAP factors are: {top_factors}."
    )
    if row.get('dem_share_2022') is not None:
        sentence += f" In 2022, Democrats won {row['dem_share_2022']*100:.1f}% of the vote here."
    elif row.get('dem_share_2024') is not None:
        sentence += f" In 2024, Democrats won {row['dem_share_2024']*100:.1f}% of the vote here."

    return sentence


def build_district_chunks():
    with open(os.path.join(PROCESSED, 'predictions_2026.json')) as f:
        predictions = json.load(f)

    chunks = []
    for row in predictions:
        state = STATE_NAMES.get(row['state_po'], row['state_po'])
        label = f"{state} district {row['district']}"
        chunks.append({'source': 'district', 'title': label, 'text': district_to_paragraph(row)})

    print('district chunks:', len(chunks), flush=True)
    return chunks


def embed_corpus(corpus, batch_size=100):
    vo = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])
    texts = [chunk['text'] for chunk in corpus]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = vo.embed(batch, model='voyage-4', input_type='document')
        all_embeddings.extend(result.embeddings)
        print(f'embedded {min(i + batch_size, len(texts))}/{len(texts)}', flush=True)
        time.sleep(0.5)

    embedding_matrix = np.array(all_embeddings)
    print('embedding matrix shape:', embedding_matrix.shape, flush=True)
    return embedding_matrix


def upload_to_supabase(corpus, embedding_matrix, batch_size=100):
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])

    rows = []
    for chunk, vector in zip(corpus, embedding_matrix):
        rows.append({
            'source': chunk['source'],
            'title': chunk['title'],
            'content': chunk['text'],
            'embedding': vector.tolist(),
        })

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table('chunks').insert(batch).execute()
        print(f'inserted {min(i + batch_size, len(rows))}/{len(rows)}', flush=True)

    print('done', flush=True)


def main():
    corpus = build_methodology_chunks() + build_district_chunks()
    print('total corpus chunks:', len(corpus), flush=True)

    embedding_matrix = embed_corpus(corpus)
    upload_to_supabase(corpus, embedding_matrix)


if __name__ == '__main__':
    main()
