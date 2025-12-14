"""Vector store helpers: creating collections and retrieving relevant embeddings.

This module isolates Chromadb usage and embedding calls to Ollama.
"""

from typing import Dict, Iterable, List, Set

import chromadb
import ollama
from tqdm import tqdm


client = chromadb.Client()


def create_vector_store(conversations: Iterable[Dict]) -> None:
    """Create/replace the `vera_conversations` collection and add embeddings.

    Conversations should be an iterable of dict-like objects with `id`,
    `prompt`, and `response` keys.
    """
    vector_store_name = "vera_conversations"
    try:
        client.delete_collection(name=vector_store_name)
    except ValueError:
        pass

    vector_store = client.create_collection(name=vector_store_name)
    for c in conversations:
        serialized_convo = f'prompt: {c["prompt"]} response: {c["response"]}'
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response['embedding']
        vector_store.add(
            ids=[str(c["id"])],
            documents=[serialized_convo],
            embeddings=[embedding],
        )


def classify_embedding(query: str, context: str) -> str:
    """Return 'yes' or 'no' indicating whether `context` matches `query`.

    Uses a small classification prompt to the LLM. Returns the model's
    normalized string result (lowercase).
    """
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly '
        'related respond "no". Do not respond "yes" unless the context is highly relevant to the search query.'
    )
    classify_convo = [
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content': 'SEARCH QUERY: What is the users name?\n\nEMBEDDED CONTEXT: You are Hoang. How can I help you today?'},
        {'role': 'assistant', 'content': 'yes'},
        {'role': 'user', 'content': 'SEARCH QUERY: Llama3 Python Voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant developed by Apple Inc.'},
        {'role': 'assistant', 'content': 'no'},
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]
    response = ollama.chat(model='llama3', messages=classify_convo)
    return response['message']['content'].strip().lower()


def retrieve_embedding(queries: List[str], results_per_query: int = 2) -> Set[str]:
    """Return a set of matching document texts for the provided queries.

    For each query we compute an embedding, query the vector store and then
    classify returned documents to ensure high relevance.
    """
    embeddings: Set[str] = set()

    for query in tqdm(queries, desc="Processing queries to vector store"):
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response['embedding']

        vector_store = client.get_collection(name="vera_conversations")
        results = vector_store.query(
            query_embeddings=[query_embedding],
            n_results=results_per_query,
        )
        best_embeddings = results['documents'][0]

        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query, context=best):
                    embeddings.add(best)

    return embeddings
