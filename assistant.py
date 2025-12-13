import ollama
import chromadb
import ast
from tqdm import tqdm
from colorama import Fore
from db import fetch_conversations, store_conversation, remove_last_conversation

client = chromadb.Client()

system_prompt = (
    'You are an AI assistant that has memory of every conversation you have ever had with this user. '
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)
convo = [{"role": "system", "content": system_prompt}]


def stream_response(prompt):
    response = ''
    stream = ollama.chat("llama3", messages=convo, stream=True)
    print(Fore.GREEN + "Vera: ")

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    print()  # For newline after completion
    store_conversation(prompt=prompt, response=response);
    convo.append({"role": "assistant", "content": response})

def create_vector_store(conversations):
    vector_store_name = "vera_conversations"
    try:
        client.delete_collection(name=vector_store_name)
    except ValueError:
        pass  # Collection does not exist, no need to delete
    vector_store = client.create_collection(name=vector_store_name)
    for c in conversations:
        serialized_convo = f'prompt: {c["prompt"]} response: {c["response"]}'
        response = ollama.embeddings(model="nomic-embed-text", prompt=serialized_convo)
        embedding = response['embedding']
        vector_store.add(
            ids=[str(c["id"])],
            documents=[serialized_convo],
            embeddings=[embedding] 
        )

def retrieve_embedding(queries, results_per_query=2):
    embeddings = set()

    for query in tqdm(queries, desc="Processing queries to vector store"):
        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        query_embedding = response['embedding']

        vector_store = client.get_collection(name="vera_conversations")
        results = vector_store.query(
            query_embeddings=[query_embedding],
            n_results=results_per_query
        )
        best_embeddings = results['documents'][0]

        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query, context=best):
                    embeddings.add(best)
    return embeddings

def create_queries(prompt):
    query_msg = (
        'You are a first principle reasoning search query AI agent. '
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have ever had with the user. With first principles create a Python list of queries to '
        'search the embeddings database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
    )
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Write an email to my car insurance company and create a pursuasive request for them to lower prices based on my good driving record'},
        {'role': 'assistant', 'content': '["What is the users name?", "What is the users current auto insurance provider?", "What is the users driving record?]'},
        {'role': 'user', 'content': 'how can i convert the speak function in my llama3 python voice assistant to use pyttsx3 instead'},
        {'role': 'assistant', 'content': '["Llama3 voice assistant", "Python voice assistant", "openAI TTS", "openai speak", "text to speech python", "convert TTS to pyttsx3"]'},
        {'role': 'user', 'content': prompt}
    ]
    response = ollama.chat(model='llama3', messages=query_convo)
    print(Fore.LIGHTYELLOW_EX + f'Vector database queries: {response["message"]["content"]} \n')
    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]

def classify_embedding(query, context):
    classify_msg = (
        'You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. '
        'You will not respond as an AI assistant. You only respond "yes" or "no". '
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly '
        'related respond "no". Do not respond "yes" unless the context is highly relevant to the search query.'
    )
    classify_convo = [
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content': f'SEARCH QUERY: What is the users name?\n\nEMBEDDED CONTEXT: You are Hoang. How can I help you today?'},
        {'role': 'assistant', 'content': 'yes'},
        {'role': 'user', 'content': f'SEARCH QUERY: Llama3 Python Voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant developed by Apple Inc.'},
        {'role': 'assistant', 'content': 'no'},
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]
    response = ollama.chat(model='llama3', messages=classify_convo)
    return response['message']['content'].strip().lower()

def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embedding(queries=queries)
    convo.append({"role": "user", "content": f'MEMORIES: {embeddings} \n USER PROMPT: {prompt}'})
    print(Fore.LIGHTYELLOW_EX + f'{len(embeddings)} relevant memories recalled added for context.\n')

conversations = fetch_conversations()
create_vector_store(conversations=conversations)
while True:
    prompt = input(Fore.CYAN + "You: ")

    # I don't like having to use recall all the time, so commenting this out for now.
    # if prompt[:7].lower() == '/recall':
    #     prompt = prompt[8:]
    #     recall(prompt=prompt)
    #     stream_response(prompt=prompt) 
    # else:
    #     convo.append({"role": "user", "content": prompt})
    #     stream_response(prompt=prompt)

    if prompt[:8].lower() == '/forget':
        remove_last_conversation()
        convo = convo[:-2]
        print("\n")
    elif prompt[:5].lower() in ['/exit', '/quit']:
        break
    else:
        recall(prompt=prompt)
        stream_response(prompt=prompt)

    
        