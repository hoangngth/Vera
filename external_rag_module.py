import os
import ollama
import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

class TwitchChatRAG:
    def __init__(self, 
                 index_path="data/rag/twitch_chat_index.faiss",
                 messages_path="data/rag/twitch_chat_messages.npy",
                 k=5,
                 max_messages=10000):
        self.index_path = index_path
        self.messages_path = messages_path
        self.k = k

        # Load existing index if exists
        if os.path.exists(index_path) and os.path.exists(messages_path):
            print("📦 Loading existing Twitch Chat FAISS index...")
            self.index = faiss.read_index(index_path)
            self.messages = np.load(messages_path, allow_pickle=True).tolist()
            print(f"✅ Loaded {len(self.messages)} messages from disk")
            return

        # Load dataset from Hugging Face
        print("Loading Twitch chat dataset...")
        dataset = load_dataset("lparkourer10/twitch_chat", split="train")
        print(f"Dataset loaded: {len(dataset)} messages")

        # Clean & limit messages
        self.messages = [
            m.lower().strip()
            for m in dataset["Message"]
            if m and not m.startswith(("!", "%", "["))
        ][:max_messages]
        
        # Create or load FAISS index
        if len(self.messages) == 0:
            raise ValueError("No messages found after cleaning!")
        
        # Create embeddings
        print("Embedding messages...")
        embeddings = []
        for msg in tqdm(self.messages, desc="Embedding messages"):
            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=msg
            )
            embeddings.append(response["embedding"])
        embeddings = np.array(embeddings).astype("float32")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print(f"FAISS index created with {len(self.messages)} messages.")

        # Optionally save index
        faiss.write_index(self.index, self.index_path)
        np.save(self.messages_path, np.array(self.messages))
        print("FAISS index and messages saved locally.")


    def retrieve(self, prompt):
        """
        Retrieve top-k relevant Twitch chat messages for a user prompt.
        Returns a list of messages.
        """
        response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
        query_embedding = response['embedding']

        D, I = self.index.search(np.array([query_embedding]).astype("float32"), self.k)
        retrieved = [self.messages[i] for i in I[0]]
        return retrieved
