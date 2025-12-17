# vera_core.py
import ollama
from db import fetch_conversations, store_conversation, remove_last_conversation
from vector_store import create_vector_store, retrieve_embedding
from query_builder import create_queries
from external_rag_module import TwitchChatRAG

system_prompt = (
    'You are Vera, an AI assistant that has memory of every conversation you have ever had with this user. '
    'On every prompt from the user, the system has checked for any relevant messages you have had with the user. '
    'If any embedded previous conversations are attached, use them for context to responding to the user, '
    'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
    'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)

twitch_rag = TwitchChatRAG(k=5, max_messages=10000)

class VeraEngine:
    def __init__(self):
        self.convo = [{"role": "system", "content": system_prompt}]
        self._init_memory()

    def _init_memory(self):
        conversations = fetch_conversations()
        try:
            create_vector_store(conversations=conversations)
        except Exception:
            pass

    def recall(self, prompt: str):
        queries = create_queries(prompt=prompt)

        embeddings = retrieve_embedding(queries=queries)
        if embeddings:
            self.convo.append({
                "role": "system",
                "content": f"Relevant memories:\n{embeddings}"
            })

        twitch_context = twitch_rag.retrieve(prompt)
        if twitch_context:
            self.convo.append({
                "role": "system",
                "content": "Twitch chat examples:\n" + "\n".join(twitch_context)
            })

        self.convo.append({"role": "user", "content": prompt})

    def generate_response(self, prompt: str):
        if prompt.lower().startswith("/forget"):
            remove_last_conversation()
            if len(self.convo) >= 2:
                self.convo.pop()
                self.convo.pop()
            return ""

        self.recall(prompt)

        response = ""
        stream = ollama.chat(
            model="llama3",
            messages=self.convo,
            stream=True
        )

        for chunk in stream:
            response += chunk["message"]["content"]

        self.convo.append({"role": "assistant", "content": response})
        store_conversation(prompt=prompt, response=response)

        return response