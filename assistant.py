import ollama
from colorama import Fore
from db import fetch_conversations, store_conversation, remove_last_conversation
from speech_to_text_whisper import listen
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


def stream_response(prompt):
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


def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embedding(queries=queries)
    convo.append({"role": "user", "content": f'MEMORIES: {embeddings} \n USER PROMPT: {prompt}'})
    print(Fore.LIGHTYELLOW_EX + f'{len(embeddings)} relevant memories recalled added for context.\n')

def handle_prompt(prompt: str):
    low = prompt.lower()

    if low.startswith('/forget'):
        remove_last_conversation()
        if len(convo) >= 2:
            convo.pop()
            convo.pop()
        print("\n")
        return

    if low.startswith('/exit') or low.startswith('/quit'):
        raise SystemExit

    recall(prompt=prompt)
    stream_response(prompt=prompt)


def main():
    conversations = fetch_conversations()
    try:
        create_vector_store(conversations=conversations)
    except Exception:
        pass

    voice_enabled = True
    voice_stream = listen() if voice_enabled else None

    while True:
        try:
            # PRIORITY: voice input
            if voice_enabled:
                prompt = next(voice_stream) # Resume the listen() function until it yields the next recognized utterance, then pause it again.
                print(Fore.CYAN + f"\nYou (voice): {prompt}")
            else:
                prompt = input(Fore.CYAN + "You: ").strip()

            if not prompt:
                continue

            handle_prompt(prompt)

        except StopIteration:
            break
        except KeyboardInterrupt:
            break
        except SystemExit:
            break


if __name__ == '__main__':
    main()