# Multi-Source Conversational RAG Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-source-conversational-rag.streamlit.app/)

A conversational AI assistant that lets you ingest documents from **PDFs, arXiv papers, web pages, and YouTube videos**, then ask natural-language questions with multi-turn memory and source citations — all running locally with no external vector database.

---

## Demo

![Demo](assets/demo.gif)

The demo shows three sources being ingested: a PDF of the paper [*"Query Rewriting for Retrieval-Augmented Large Language Models"* (Ma et al., 2023)](https://arxiv.org/abs/2305.14283), an [AWS web article on RAG](https://aws.amazon.com/what-is/retrieval-augmented-generation/), and an [IBM YouTube video explaining RAG](https://www.youtube.com/watch?v=qppV3n3YlF8). Re-ranking is enabled, and two questions are asked — the second is a follow-up that triggers automatic query rewriting. The rewritten query and per-source citations are visible in the UI.

---

## Features

| Feature | Details |
|---|---|
| **Multi-source ingestion** | PDF upload · arXiv paper (by ID or URL) · Web URL (trafilatura) · YouTube transcript (youtube-transcript-api) |
| **Semantic search** | Sentence-transformer embeddings (`all-MiniLM-L6-v2`) stored in local ChromaDB |
| **Source diversity** | Per-source retrieval cap ensures multiple sources contribute to every answer |
| **Conversational memory** | Follow-up questions resolved via LLM query rewriting before retrieval |
| **Source citations** | Numbered `[1]`, `[2]` inline citations with a collapsible Sources expander; YouTube links are timestamped |
| **Re-ranking (opt-in)** | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores retrieved chunks — runs locally, no API needed |
| **Live pipeline display** | Step-by-step progress card (Rewrite → Retrieve → Generate) visible while the model thinks |
| **Export conversation** | Download the full chat as a PDF with one click |

---

## Architecture

```mermaid
flowchart TD
    subgraph Sources
        A[📄 PDF] --> C
        B[🌐 Web URL] --> C
        D[▶️ YouTube] --> C
        X[📜 arXiv] --> C
        C[Chunker] --> E[Embedder\nall-MiniLM-L6-v2]
        E --> F[(ChromaDB\nlocal)]
    end

    subgraph Query
        G[User question] --> H{History?}
        H -- Yes --> I[LLM rewrite\nstandalone question]
        H -- No --> J
        I --> J[Retriever\ntop-k chunks]
        J -- opt-in --> K[Cross-encoder\nre-rank]
        K --> L
        J --> L[LLM Generator\nGWDG API]
        L --> M[Answer + citations]
    end

    F --> J
    
    %% Give the edge an id to style it
    linkStyle 14 stroke:green,stroke-width:2px
```

**Data flow:**

1. **Write path** — Source → Chunker (500 chars, 50 overlap) → Embedder → ChromaDB
2. **Read path** — Query → *(optional) LLM rewrite* → Retriever → *(optional) cross-encoder re-rank* → LLM Generator → Answer with citations

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- A GWDG API key (or any OpenAI-compatible endpoint)

### Installation

```bash
git clone https://github.com/<your-username>/multi-source-conversational-rag.git
cd multi-source-conversational-rag

cp .env.example .env          # fill in your API credentials
make install                  # creates .venv and installs all dependencies
make run                      # launches the Streamlit app at http://localhost:8501
```

> **First run:** the embedding model (`all-MiniLM-L6-v2`, ~90 MB) and re-ranking model (`ms-marco-MiniLM-L-6-v2`, ~90 MB) are downloaded from HuggingFace on first use and cached locally. Expect a one-time wait of 1–2 minutes on a typical connection.

### Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `GWDG_API_KEY` | ✅ | API key for the GWDG (or any OpenAI-compatible) LLM endpoint |
| `GWDG_API_BASE` | ✅ | Base URL, e.g. `https://chat-ai.academiccloud.de/v1` |
| `GWDG_MODEL_NAME` | ✅ | Model name, e.g. `meta-llama-3.1-70b-instruct` |
| `HF_TOKEN` | ☑️ optional | HuggingFace token — only needed to access gated models. The embedding and re-ranking models used here are public, so this can be left empty. |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | ☑️ optional | Set to `1` on Windows to suppress a cosmetic HuggingFace cache warning (no functional impact). |

---

## Development

```bash
make lint                                    # ruff check + format
uv run pytest tests/                         # run all tests (80 tests, no API calls)
uv run pytest tests/test_ingestion.py        # run a single file
uv run python scripts/reset_vectorstore.py   # wipe ChromaDB and start fresh
```

---

## Project Structure

```
├── app/
│   ├── app.py                  # Streamlit entry point
│   ├── page_config.py          # page title / icon / layout constants
│   └── components/
│       ├── chat.py             # chat UI, citation rendering, pipeline progress card
│       ├── sidebar.py          # source ingestion forms + retrieval settings
│       └── source_viewer.py    # ingested sources list with delete buttons
│
├── rag/
│   ├── ingestion/
│   │   ├── base.py             # Document dataclass + Ingestor ABC
│   │   ├── pdf_ingestor.py     # PyMuPDF — one Document per page
│   │   ├── web_ingestor.py     # trafilatura — article extraction
│   │   ├── youtube_ingestor.py # youtube-transcript-api — timestamp-bounded chunks
│   │   └── arxiv_ingestor.py   # downloads arXiv PDF by ID or URL → delegates to PDFIngestor
│   ├── chunking/chunker.py     # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings/embedder.py  # sentence-transformers bi-encoder
│   ├── vectorstore/chroma_store.py
│   ├── retrieval/retriever.py  # top-k + per-source cap + optional cross-encoder rerank
│   ├── generation/generator.py # LangChain → GWDG LLM, citation prompt
│   ├── memory/conversation.py  # query rewriting with conversation history
│   ├── models.py               # Citation + Answer dataclasses, citation regex
│   └── pipeline.py             # orchestrator — exposes ingest / ask / granular steps
│
├── config/settings.py          # all tunables in one place (reads .env + st.secrets)
├── tests/                      # 80 pytest tests, all offline (LLM and HTTP mocked)
├── .streamlit/
│   ├── config.toml             # theme + server settings
│   └── secrets.toml.example    # Streamlit Cloud secrets template
├── requirements.txt            # pinned deps for Streamlit Cloud (generated by uv)
└── pyproject.toml              # project metadata + ruff config
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit 1.39+ |
| LLM integration | LangChain + langchain-openai |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Re-ranking | sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector store | ChromaDB (local, file-persisted) |
| PDF parsing | PyMuPDF |
| arXiv ingestion | requests (direct PDF download) |
| Web extraction | trafilatura |
| YouTube transcripts | youtube-transcript-api |
| PDF export | fpdf2 |
| Linting / formatting | Ruff |
| Testing | pytest |
| Package management | uv |

---

## Possible Future Improvements

| Improvement | How | Effort | Cost |
|---|---|---|---|
| **Hybrid search** (BM25 + vector) | Add `rank_bm25` for keyword retrieval; merge scores with reciprocal rank fusion before the cross-encoder step | Medium (2–3 days) | Free — local |
| **More source types** (Notion, Google Docs) | `notion-client` (Notion API token); `google-api-python-client` (OAuth2). Each is a new `Ingestor` subclass | Medium per source | Free tiers available; Google Docs requires OAuth setup |
| **Streaming LLM responses** | Replace `generator.generate()` with LangChain's `stream()`; render token-by-token with Streamlit's `st.write_stream()` (v1.31+). Citation marker parsing must be deferred to stream end | Low–Medium (1–2 days) | Free |
| **User authentication** | Streamlit Community Cloud has built-in viewer auth (Google/GitHub). For custom auth: `streamlit-authenticator`. Full multi-user data isolation requires per-user ChromaDB collections | High — significant architecture change | Streamlit Cloud free tier supports viewer auth |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

[**Farmand Bazdiditehrani**](https://www.linkedin.com/in/farmand-bt/) · M.Sc. in Management & Data Science · [farmand.bazdiditehrani@gmail.com](mailto:farmand.bazdiditehrani@gmail.com)
