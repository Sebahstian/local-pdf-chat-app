# 📄 PDF Chat — talk to any PDF, 100% locally

Upload a PDF and have a conversation with it — follow-up questions work, and every
answer cites the pages it came from. Everything runs on your machine: local LLM
(**Ollama + Llama 3.2**), local embeddings, in-memory vector search. **No API keys,
no cloud, your documents never leave your computer.**

![Demo](assets/demo.gif)

**Stack:** Python · LangChain (1.x, LCEL) · Streamlit · Ollama (`llama3.2:3b` + `nomic-embed-text`)

## Quickstart

### Option A — native (macOS/Linux, ~2 min)

Prereqs: [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) installed.

```bash
git clone https://github.com/YOUR_USERNAME/local-pdf-chat-app && cd local-pdf-chat-app
make setup          # install Python deps
make pull-models    # download llama3.2:3b + nomic-embed-text (~2.3 GB, once)
make run            # opens http://localhost:8501
```

### Option B — Docker (nothing but Docker required)

```bash
docker compose up   # first run downloads models (~2.3 GB), then serves :8501
```

Then open <http://localhost:8501>, upload a PDF, and ask away.

## How it works

```
                 ┌─────────────────────────── Streamlit UI (app.py) ────────────────────────────┐
                 │  sidebar: PDF uploader        main: chat history + st.chat_input (streaming) │
                 └───────────────┬───────────────────────────────────────┬─────────────────────┘
                                 │ on upload                             │ on question
                                 ▼                                       ▼
   pdf_processor.py          vectorstore.py                       chain.py (LCEL)
   pypdf ─────► Recursive ─► OllamaEmbeddings ─► InMemory        1. condense question
   (page meta)  TextSplitter (nomic-embed-text)  VectorStore        against chat history
                                                 (top-k) ◄────── 2. retrieve top-k chunks
                                                                 3. answer from context only
                                                                        │
                                                                 ChatOllama (llama3.2:3b,
                                                                  streamed tokens)
                                                                        │
                                                            answer + source page citations
```

1. **Ingest** — the PDF is parsed page-by-page (`pypdf`), keeping page numbers as
   metadata, then split into ~1000-char overlapping chunks.
2. **Index** — chunks are embedded locally with `nomic-embed-text` and stored in an
   in-memory vector index for the session.
3. **Ask** — if you're mid-conversation, your question is first rewritten into a
   standalone query (so *"and after that?"* retrieves the right chunks), then the
   top-k chunks are fetched.
4. **Answer** — `llama3.2:3b` answers **only from the retrieved context** (it says
   "I don't know" rather than inventing things), streams token-by-token, and the
   pages of the retrieved chunks are shown as citations.

## Design decisions

- **Local-first** — a privacy-preserving RAG pipeline with zero external services:
  useful for confidential documents, and free to run.
- **Modern LangChain (1.x)** — the pipeline is composed from LCEL primitives
  (`prompt | llm | parser`) in [`src/chain.py`](src/chain.py), not the deprecated
  `ConversationalRetrievalChain` or the sunset `langchain-community` package.
- **Real citations** — page metadata flows from the PDF parser through chunking and
  retrieval to the UI, so cited pages are the actual retrieval sources, never
  model-generated.
- **In-memory vector store** — one PDF per session is a small corpus; exact search in
  `langchain-core`'s `InMemoryVectorStore` is correct and dependency-free. Persistence
  (Chroma/FAISS) would be a change isolated to [`src/vectorstore.py`](src/vectorstore.py).
- **Config over code** — models and retrieval knobs live in [`.env`](.env.example);
  e.g. switch to a bigger model with `LLM_MODEL=qwen3:8b`, no code change.

## Project layout

```
app.py                  Streamlit UI: upload, chat loop, streaming, citations
src/config.py           env-driven settings (models, chunking, top-k)
src/pdf_processor.py    PDF → per-page Documents → chunks (page metadata preserved)
src/vectorstore.py      local embeddings + in-memory vector index
src/chain.py            LCEL pipeline: condense → retrieve → grounded answer
tests/                  unit tests for the ingestion pipeline (no Ollama required)
```

## Tests

```bash
make test
```

The ingestion pipeline (parsing, chunking, metadata/citation plumbing) is covered by
pure unit tests — the fixture PDF is generated in-memory, and no model has to be running.
