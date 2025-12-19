import chromadb
import ollama
from tqdm import tqdm

client = chromadb.Client()


def create_vector_store(conversations):
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


def classify_embedding(query, context):
    classify_msg = (
        "You are an embedding classification AI agent. "
        "Your input will be a search query and one chunk of embedded text. "
        "You will NOT respond as an AI assistant. You will only respond with the single word 'yes' or 'no'. "
        "Determine whether the embedded context directly contains information needed to answer the search query. "
        "Respond 'yes' only if the context is highly relevant and directly useful to the query. "
        "If it is not directly relevant, respond 'no'. "
        "Do not explain your answer and do not output anything other than 'yes' or 'no'."
    )

    classify_convo = [
        {"role": "system", "content": classify_msg},

        # Example 1: directly relevant
        {"role": "user", "content": "SEARCH QUERY: What is the user's name?\n\nEMBEDDED CONTEXT: You are Hoang. How can I help you today?"},
        {"role": "assistant", "content": "yes"},

        # Example 2: not relevant
        {"role": "user", "content": "SEARCH QUERY: Llama3 Python Voice Assistant\n\nEMBEDDED CONTEXT: Siri is a voice assistant developed by Apple Inc."},
        {"role": "assistant", "content": "no"},

        # Dynamic query/context
        {"role": "user", "content": f"SEARCH QUERY: {query}\n\nEMBEDDED CONTEXT: {context}"}
    ]
    
    response = ollama.chat(model='llama3', messages=classify_convo)
    return response['message']['content'].strip().lower()


def retrieve_embedding(queries, results_per_query=2):
    embeddings = set()

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
