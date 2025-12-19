# vera_core.py
import ollama
from db import fetch_conversations, store_conversation, remove_last_conversation
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

twitch_rag = TwitchChatRAG(k=5, max_messages=10000)

class VeraEngine:
    def __init__(self):
        self.temp_context = []
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

        # Memory RAG
        embeddings = retrieve_embedding(queries=queries)
        if embeddings:
            self.temp_context.append({
                "role": "system",
                "content": f"Use the following relevant memories to respond intelligently, but do not repeat verbatim:\n{embeddings}"
            })

        # Twitch Chat RAG
        twitch_context = twitch_rag.retrieve(prompt)
        if twitch_context:
            self.temp_context.append({
                "role": "system",
                "content": "Use these Twitch chat examples as reference for style and context, but do not repeat verbatim:\n" + "\n".join(twitch_context)
            })


    def generate_response(self, prompt: str):
        if prompt.lower().startswith("/forget"):
            remove_last_conversation()
            if len(self.convo) >= 2:
                self.convo.pop()
                self.convo.pop()
            return ""

        self.recall(prompt)

        response = ''
        full_context = self.convo + self.temp_context + [{"role": "user", "content": prompt}]
        stream = ollama.chat("llama3", messages=full_context, stream=True)

        for chunk in stream:
            response += chunk["message"]["content"]

        # Reset temp_context
        self.temp_context.clear()

        # Store conversation 
        store_conversation(prompt=prompt, response=response)
        self.convo.append({"role": "user", "content": prompt})
        self.convo.append({"role": "assistant", "content": response})

        return response