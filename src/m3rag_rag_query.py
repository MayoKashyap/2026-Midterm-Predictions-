import os

import anthropic
import voyageai
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(os.path.expanduser('~/Documents/2026MLpredictor/.env'))

GENERATION_MODEL = 'claude-sonnet-5'


def get_clients():
    vo = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])
    supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    claude = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    return vo, supabase, claude


def retrieve(vo, supabase, question, match_count=5):
    question_embedding = vo.embed([question], model='voyage-4', input_type='query').embeddings[0]
    result = supabase.rpc('match_chunks', {
        'query_embedding': question_embedding,
        'match_count': match_count,
    }).execute()
    return result.data


def generate_answer(claude, question, top_chunk):
    prompt = f"""Context: {top_chunk}

Question: {question}

Answer using only the context above. If the context doesn't contain the answer, say you don't have that information."""

    response = claude.messages.create(
        model=GENERATION_MODEL,
        max_tokens=300,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.content[0].text


def ask(question, match_count=5):
    vo, supabase, claude = get_clients()

    matches = retrieve(vo, supabase, question, match_count=match_count)
    for row in matches:
        print(f"{row['similarity']:.3f}  [{row['source']}] {row['title']}", flush=True)

    top_chunk = matches[0]['content']
    answer = generate_answer(claude, question, top_chunk)
    print(flush=True)
    print(answer, flush=True)
    return answer


if __name__ == '__main__':
    ask("Why is Alabama's 1st congressional district rated Safe R?")
