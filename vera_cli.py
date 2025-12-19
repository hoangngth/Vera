import ollama
from colorama import Fore
from db import fetch_conversations, store_conversation, remove_last_conversation
from speech_to_text_whisper import listen, clear_audio_queue
from text_to_speech_xtts import speak
from vector_store import create_vector_store, retrieve_embedding
from query_builder import create_queries
from external_rag_module import TwitchChatRAG


system_prompt = (
    "You are Vera, an AI assistant with memory of past conversations with this user. "
    "Always focus on the current user prompt. "
    "Use past conversation context only if it is directly relevant. "
    "Do not mention or explain that you are recalling past conversations. "
    "Do not repeat past conversations verbatim. "
    "Respond naturally, clearly, and helpfully, using any relevant past information only when it is truly useful."
)

agent_voice_enabled = False
user_voice_enabled = True

temp_context = []
convo = [{"role": "system", "content": system_prompt}]

twitch_rag = TwitchChatRAG(
    k=5,
    max_messages=10000
)

def stream_response(prompt):
    response = ''

    full_context = convo + temp_context + [{"role": "user", "content": prompt}]
    stream = ollama.chat("llama3", messages=full_context, stream=True)
    print(Fore.GREEN + "Vera: ")

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    print("\n")

    # Reset temp_context
    temp_context.clear()

    # Store conversation 
    store_conversation(prompt=prompt, response=response)
    convo.append({"role": "user", "content": prompt})
    convo.append({"role": "assistant", "content": response})

    # Speak
    if agent_voice_enabled and response.strip():
        speak(response, filename="response.wav")


def recall(prompt):
    queries = create_queries(prompt=prompt)

    # Memory RAG
    embeddings = retrieve_embedding(queries=queries)
    if embeddings:
        temp_context.append({
            "role": "system",
            "content": f"Use the following relevant memories to respond intelligently, but do not repeat verbatim:\n{embeddings}"
        })

    # Twitch Chat RAG
    twitch_context = twitch_rag.retrieve(prompt)
    if twitch_context:
        temp_context.append({
            "role": "system",
            "content": "Use these Twitch chat examples as reference for style and context, but do not repeat verbatim:\n" + "\n".join(twitch_context)
        })

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
    stream_response(prompt)


def main():
    conversations = fetch_conversations()
    try:
        create_vector_store(conversations=conversations)
    except Exception:
        pass

    global user_voice_enabled
    try:
        if user_voice_enabled:
            next(listen())
            print("🎤 Voice input enabled")
    except Exception as e:
        print(f"⚠️ Voice input unavailable, falling back to text: {e}")
        user_voice_enabled = False

    while True:
        try:
            if user_voice_enabled:
                clear_audio_queue()
                prompt = next(listen()) # Resume the listen() function until it yields the next recognized utterance, then pause it again.
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