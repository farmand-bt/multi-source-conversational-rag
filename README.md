# Multi-Source Conversational RAG Assistant

Conversational RAG assistant that lets you ingest documents from PDFs, web URLs, and YouTube transcripts, then ask natural-language questions with multi-turn memory and source citations.

## What works now

- **PDF ingestion** — upload a PDF from the sidebar; pages are extracted, chunked, and embedded into a local ChromaDB vector store.
- **Web page ingestion** — paste any URL; trafilatura extracts clean article text.
- **YouTube ingestion** — paste a YouTube URL; the transcript is fetched and grouped into timestamp-bounded chunks.
- **Q&A with citations** — ask a question; the top-5 most relevant chunks are retrieved and sent to the LLM, which answers with typed citations — `[PDF: name, page N]`, `[Web: title, URL]`, or `[YouTube: title, MM:SS]` — rendered in a collapsible expander.
- **Source management** — view all ingested sources with type-specific metadata (page count, domain, video URL) and delete them individually.

Multi-turn memory and query rewriting are coming in Milestone 5.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install it with `pip install uv` or see the uv docs for other options.

```bash
cp .env.example .env   # fill in your GWDG API credentials
make install           # uv creates .venv and installs all dependencies
make run               # launches the Streamlit app
```

## Other commands

```bash
make lint                                    # ruff check + format
uv run pytest tests/                         # run all tests
uv run pytest tests/test_ingestion.py        # run a single test file
uv run python scripts/reset_vectorstore.py   # wipe ChromaDB and start fresh
```

## Environment variables

| Variable | Description |
|---|---|
| `GWDG_API_KEY` | API key for the GWDG LLM endpoint |
| `GWDG_API_BASE` | Base URL of the OpenAI-compatible endpoint |
| `GWDG_MODEL_NAME` | Model name to use for generation |

## Tech stack

Python 3.10+ · LangChain · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · GWDG API · PyMuPDF · trafilatura · youtube-transcript-api · Streamlit · Ruff
