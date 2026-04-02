# 🎬 yt-transcript-ai

Ask questions about any YouTube video — entirely offline, entirely local.

Fetches the transcript from a YouTube video, chunks it up, embeds it into a FAISS vector store, and uses a local LLM via [Ollama](https://ollama.com) to answer your questions. No API keys. No cloud. Just you and your GPU.

> Inspired by the Coursera GenAI with LangChain course, rebuilt from scratch with full Ollama integration for local-first usage.

---

## How it works

```
YouTube URL
    │
    ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Transcript   │────▶│  Text Splitter  │────▶│  FAISS Index  │
│  Fetcher      │     │  (chunking)     │     │  (embeddings) │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                    │
                                    user question ──┤
                                                    ▼
                                          ┌──────────────────┐
                                          │  LangChain QA     │
                                          │  Chain (Ollama)    │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                                Answer
```

## Features

- **Transcript extraction** — auto-fetches English transcripts (prefers manual over auto-generated)
- **RAG pipeline** — chunks transcript → embeds with Ollama → stores in FAISS → retrieves relevant context
- **Summarization** — get a concise summary of any video
- **Q&A** — ask specific questions and get grounded answers from the video content
- **100% local** — runs entirely on your machine via Ollama

## Project structure

```
├── main.py                  # Entry point
├── core/
│   └── yt.py                # Transcript fetching, processing, orchestration
├── llm/
│   └── llm_client.py        # Ollama LLM & embedding model setup
├── rag/
│   ├── textsplitter.py      # Recursive text chunking
│   ├── faiss.py             # FAISS index creation, similarity search, answer generation
│   └── vectorstore.py       # Vector store utilities
├── prompt/
│   └── template.py          # LangChain prompt templates (summary & QA)
├── chain/
│   └── chain_builder.py     # LangChain chain construction
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally

### Install

```bash
git clone https://github.com/<your-user>/yt-transcript-ai.git
cd yt-transcript-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pull the models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text   # or bge-m3
```

### Configure (optional)

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TEMPERATURE=0.5
EMBEDDING_MODEL=nomic-embed-text
```

## Usage

```bash
python main.py
```

## Tech stack

| Component | Tool |
|-----------|------|
| LLM | Ollama (llama3.2) |
| Embeddings | Ollama (nomic-embed-text / bge-m3) |
| Orchestration | LangChain |
| Vector store | FAISS |
| Transcripts | youtube-transcript-api |
