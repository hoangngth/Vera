Vera API 🤖

Vera is a personal-use AI assistant, designed for local experimentation and future evolution into a potential SaaS product.

It features:

Long-term conversation memory (vector store)

Retrieval-Augmented Generation (RAG)

Streaming LLM responses via Ollama

Optional speech-to-text and text-to-speech modules

Secure public access via Cloudflare Tunnel (free)

This repository exposes Vera as an HTTP API, allowing usage from:

Postman

Web applications (Vue / React)

Mobile applications (iOS / Android)

Any internet-connected client

✨ Features

🧠 Conversation Memory – semantic recall of relevant past interactions

🔍 RAG Pipeline – vector search + external context (e.g. Twitch chat)

💬 Chat API – simple and extensible /chat endpoint

🖥️ CLI Interface – legacy local assistant for fast iteration

🗣️ Speech-to-Text – Whisper (default), Vosk (offline fallback)

🔊 Text-to-Speech – XTTS (disabled by default due to latency)

🌐 Internet Exposure – free HTTPS access via Cloudflare Tunnel

🔐 Optional Authentication – API key–based security

🏗️ Project Structure
.
├── api.py                     # FastAPI entrypoint (HTTP API)
├── vera_core.py               # Core Vera engine (LLM + memory + RAG)
├── vera_cli.py                # Legacy CLI interface for local testing
├── authorization.py           # Optional API key authentication
├── db.py                      # Conversation persistence
├── vector_store.py            # Embeddings + vector database logic
├── query_builder.py           # Query expansion logic
├── external_rag_module.py     # External RAG sources (e.g. Twitch)
├── speech_to_text_whisper.py  # Whisper STT (default)
├── speech_to_text_vosk.py     # Vosk STT (offline fallback)
├── text_to_speech_xtts.py     # XTTS TTS (optional)
├── schema.sql                 # Database schema (table creation)
├── requirements.txt
└── README.md

⚙️ Requirements

Python 3.10+

Ollama installed and running

Supported LLM model (e.g. llama3)

📦 Installation
pip install -r requirements.txt


Ensure Ollama is running:

ollama serve


Pull the model:

ollama pull llama3

▶️ Run the API Locally
uvicorn api:app --host 0.0.0.0 --port 8000


API available at:

http://localhost:8000


Swagger UI:

http://localhost:8000/docs

🧪 Test with Postman
Endpoint

POST /chat

Request Body
{
  "message": "Hello Vera"
}

Response
{
  "session_id": "uuid",
  "response": "Hello! How can I help you today?"
}

🖥️ Run Vera via CLI (Legacy)

Before the API existed, Vera was used as a local CLI assistant for rapid testing.

python vera_cli.py


This mode supports:

Local conversation loop

Memory recall

Optional voice input/output

⚠️ The CLI is considered legacy and intended mainly for development/debugging.

🌍 Expose API to the Internet (FREE)
cloudflared tunnel --url http://localhost:8000


Example public URL:

https://random-name.trycloudflare.com


No domain or payment required.

🔐 Security

Authentication is optional by design.

Example header:

Authorization: Bearer YOUR_API_KEY

🔊 Voice (Optional)
Speech-to-Text

Whisper (default)

Vosk (offline fallback)

Text-to-Speech

XTTS (disabled by default)

Outputs .wav files

🧠 Architecture
Client (API / CLI)
        |
        v
     FastAPI
        |
        v
   VeraEngine
        |
        +--> Memory (Vector Store)
        |
        +--> External RAG
        |
        v
      LLM (Ollama)

🗄️ Database

Schema defined in schema.sql.

❤️ Credits

Built by Hoang Nguyen The