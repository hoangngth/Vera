"""Main CLI assistant loop and helpers.

This module handles the interactive loop, streaming responses, and
recalling relevant memories via the vector store.
"""

import ollama
from colorama import Fore
from db import fetch_conversations, store_conversation, remove_last_conversation
from vector_store import create_vector_store, retrieve_embedding
from query_builder import create_queries


system_prompt = (
    'You are an AI assistant that has memory of every conversation you have ever had with this user. '
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)

convo = [{"role": "system", "content": system_prompt}]


def stream_response(prompt: str) -> None:
    """Stream model output for a user prompt and persist the conversation.

    Appends the assistant response to the in-memory `convo` and stores the
    conversation in the database using `store_conversation`.
    """
    response = ''
    stream = ollama.chat("llama3", messages=convo, stream=True)
    print(Fore.GREEN + "Vera: ")

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    print()

    store_conversation(prompt=prompt, response=response)
    convo.append({"role": "assistant", "content": response})


def recall(prompt: str) -> None:
    """Generate queries for the prompt and retrieve matching memories.

    The recalled memories are appended to `convo` as a context message.
    """
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embedding(queries=queries)
    convo.append({"role": "user", "content": f'MEMORIES: {embeddings} \n USER PROMPT: {prompt}'})
    print(Fore.LIGHTYELLOW_EX + f'{len(embeddings)} relevant memories recalled added for context.\n')


def main() -> None:
    """Run the interactive assistant loop."""
    conversations = fetch_conversations()
    try:
        create_vector_store(conversations=conversations)
    except Exception:
        # Creating the vector store is non-fatal for the CLI; continue.
        pass

    while True:
        prompt = input(Fore.CYAN + "You: ").strip()
        if not prompt:
            continue

        low = prompt.lower()
        if low.startswith('/forget'):
            remove_last_conversation()
            # remove the last user+assistant entries if present
            if len(convo) >= 2:
                convo.pop()
                convo.pop()
            print("\n")
            continue

        if low.startswith('/exit') or low.startswith('/quit'):
            break

        recall(prompt=prompt)
        stream_response(prompt=prompt)


if __name__ == '__main__':
    main()


