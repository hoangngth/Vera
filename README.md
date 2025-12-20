# Vera 🤖

Vera is a **personal-use AI assistant**, designed for **local experimentation** and future evolution into a potential **SaaS product**.

It features:
- Long-term conversation memory (vector store)
- Retrieval-Augmented Generation (RAG)
- Streaming LLM responses via Ollama
- Optional speech-to-text and text-to-speech modules

This repository exposes Vera as an **HTTP API**, allowing usage from:
- Postman
- Web applications (Vue / React)
- Mobile applications (iOS / Android)
- Any internet-connected client

---

## ✨ Features

- 🧠 **Conversation Memory** – semantic recall of relevant past interactions  
- 🔍 **RAG Pipeline** – vector search + external context (e.g. Twitch chat)  
- 💬 **Chat API** – simple and extensible `/chat` endpoint  
- 🖥️ **CLI Interface** – legacy local assistant for fast iteration  
- 🗣️ **Speech-to-Text** – Whisper (default), Vosk (offline fallback)  
- 🔊 **Text-to-Speech** – XTTS (disabled by default due to latency)  
- 🌐 **Internet Exposure** – free HTTPS access via Cloudflare Tunnel  
- 🔐 **Optional Authentication** – API key–based security  

---

## 🏗️ Project Structure

```text
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
```

## ⚙️ Requirements

- Python **3.10+**
- Ollama installed and running
- Supported LLM model (e.g. `llama3`)

---

## 📦 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure Ollama is running:
```bash
ollama serve
```

Pull the model:
```bash
ollama pull llama3
```

## ▶️ Run the API Locally
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API available at:

http://localhost:8000

Swagger UI:

http://localhost:8000/docs

## Endpoints

### 1. `POST /chat` — Text Chat

Send a text message to Vera and receive a response.

**Request Body:**

| Field        | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `session_id` | string | No       | Optional session ID to continue a previous conversation. If omitted, a new session is created. |
| `message`    | string | Yes      | The user message or prompt for Vera. |

**Example Request (JSON):**

```json
{
  "session_id": "abc123",
  "message": "Hello Vera, how are you?"
}
```

**Reponse:**

| Field        | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `session_id` | string | No       | The session ID (useful to continue the conversation). |
| `response`    | string | Yes      | Vera's generated response to the message. |

**Example Response (JSON):**

```json
{
  "session_id": "abc123",
  "response": "Hello! I'm doing great. How can I assist you today?"
}
```

### 2. `POST /transcribe` — Audio Transcription

Upload an audio file to transcribe its content into text.

**Request:**

- **Content-Type:** `multipart/form-data`
- **Form Field:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file to transcribe. Supported formats: any audio MIME type. |

**Example (curl):**

```bash
curl -X POST "http://<your-server-address>/transcribe" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -F "file=@audio_sample.webm"
```

| Field | Type  | Description |
|-------|-------|-------------|
| `transcript` | string | Transcribed text from the audio file. |

**Example Response (JSON):**

```json
{
  "transcript": "Hello Vera, can you tell me a fun fact?"
}
```

### 3. `POST /audio` — Audio Chat

Upload an audio file and receive both transcription and Vera's response.

**Request:**

- **Content-Type:** `multipart/form-data`
- **Form Fields:**

| Field        | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `file`       | file   | Yes      | Audio file containing the user's speech. |
| `session_id` | string | No       | Optional session ID for conversation continuity. |

**Example (curl):**

```bash
curl -X POST "http://<your-server-address>/audio" \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -F "file=@user_speech.webm" \
  -F "session_id=abc123"
  ```

**Response:**

| Field               | Type           | Description |
|--------------------|----------------|-------------|
| `session_id`        | string         | Session ID for conversation continuity. |
| `transcript`        | string         | Transcribed text from the audio. |
| `response`          | string         | Vera's generated response to the transcribed text. |
| `response_audio_url`| string or null | Placeholder for TTS audio URL (currently `null`). |

**Example Response:**

```json
{
  "session_id": "abc123",
  "transcript": "Hello Vera, tell me a joke.",
  "response": "Sure! Why did the computer go to the doctor? Because it caught a virus!",
  "response_audio_url": null
}

## 🖥️ Run Vera via CLI (Legacy)

Before the API existed, Vera was used as a **local CLI assistant** for rapid testing.

```bash
python vera_cli.py
```

This mode supports:

- Local conversation loop
- Memory recall
- Optional voice input/output

⚠️ The CLI is considered **legacy** and intended mainly for development/debugging.

## 🔐 Security

Authentication is **optional by design.**

### Example header:

Authorization: Bearer YOUR_API_KEY

## 🔊 Voice (Optional)

### Speech-to-Text

- Whisper (default)
- Vosk (offline fallback)

### Text-to-Speech

- XTTS (disabled by default)
- Outputs .wav files

## 🧠 Architecture
```plaintext
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
```

## 🗄️ Database
Schema defined in schema.sql.

## ❤️ Credits
Built by Hoang Nguyen The