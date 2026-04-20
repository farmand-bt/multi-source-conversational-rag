# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Milestones complete:** M1 (scaffolding), M2 (PDF ingestion pipeline), M3 (basic RAG query), M4 (multi-source ingestion), M5 (conversational memory), M6 (polish, README, deployment). Additional post-M6 features: arXiv ingestion, YouTube transcript translation, PDF export.

**M2 notes:**
- `app/config.py` was deleted; page constants live in `app/page_config.py` to avoid shadowing the `config/` settings package (Streamlit prepends `app/` to `sys.path`).
- Imports inside `app/app.py` that reference other files in `app/` must use **bare names** (`from components.sidebar import ...`, `from page_config import ...`), NOT package-qualified names (`from app.components...`). Streamlit adds `app/` to `sys.path[0]`, causing `app.py` itself to be seen as module `app` (a module, not a package), which makes `from app.components` fail. This is intentional — `app/app.py` is only ever launched via `streamlit run`.
- **Per-user session isolation:** `Embedder` is `@st.cache_resource` (shared across users — no state). `RAGPipeline` is created per-session in `st.session_state` with `ephemeral=True` AND a UUID collection name (`f"session_{uuid.uuid4().hex}"`). **Critical:** ChromaDB 1.x (Rust backend) shares in-process memory between `EphemeralClient()` instances — all instances using the same collection name `"documents"` would see each other's data and cause `NotFoundError` when one user calls "Delete all" (deletes the shared collection, leaving other users with stale collection-ID references). The UUID collection name gives each session a private namespace within the shared Rust store. Do NOT cache the pipeline as `@st.cache_resource` and do NOT use a fixed collection name for ephemeral sessions.
- `ChromaStore` receives pre-computed embeddings from `Embedder` (embedding path) — `Embedder` is the single embedding authority; ChromaDB has no internal embedding function.
- Metadata `None` values are stripped before passing to ChromaDB (ChromaDB rejects `None` metadata values).

**M3 notes:**
- `rag/models.py` defines two frozen dataclasses: `Citation(source_name, location)` and `Answer(text, citations)`. `Answer.from_raw()` parses `[Source: name, location]` markers from raw LLM output using a regex.
- `Generator` is **lazy-initialized** in `RAGPipeline` (`self._generator: Generator | None = None`) — it is only constructed on the first `ask()` call, so the app starts without error even when GWDG credentials are missing.
- `ChromaStore.query()` now returns `list[tuple[Document, float]]` — similarity score is `round(1.0 - cosine_distance, 4)`, in range [0, 1].
- `tests/conftest.py` provides a **session-scoped `embedder` fixture** so the sentence-transformer model is loaded only once across the entire test suite.
- `Generator.generate()` accepts `history` so conversation context is passed to the LLM.
- Optional re-ranking is deferred; `Retriever.retrieve()` returns results already ordered by cosine similarity.

**M4 notes:**
- `WebIngestor` uses `trafilatura.fetch_url` + `trafilatura.extract` (with `favor_recall=True`) and `extract_metadata` for the page title. Returns one `Document` per URL. Source ID = SHA-256 of the URL.
- `YouTubeIngestor` uses `youtube-transcript-api` v1.2.4 instance API: `YouTubeTranscriptApi().fetch(video_id)`. Video title is fetched from the YouTube oEmbed endpoint (no API key needed) with a silent fallback to the video ID. Transcripts are pre-grouped into ~`CHUNK_SIZE` character blocks preserving segment timestamps.
- Citation format changed from `[Source: name, location]` to type-prefixed tags: `[PDF: name, page N]` / `[Web: title, URL]` / `[YouTube: title, timestamped_URL]`. `rag/models.py` regex and `Citation` dataclass updated accordingly (`Citation` now has a `source_type` field).
- For YouTube, the context header sent to the LLM includes the full timestamped URL (`https://youtube.com/watch?v=ID&t=Ns`) as the location field, so the LLM can copy it verbatim. The helper `_ts_to_seconds()` in `generator.py` converts `MM:SS`/`H:MM:SS` to seconds.
- The sidebar has separate 📄 PDF / 🌐 Web Page / ▶️ YouTube sections, each with an Ingest button and a Clear button. Clear works via a session-state counter key — incrementing it forces Streamlit to render a new empty `text_input` on the next rerun (direct session-state writes to a widget key raise `StreamlitAPIException`).
- `app/components/chat.py` replaces inline `[PDF/Web/YouTube: …]` markers with numbered `[1]`, `[2]` references (coloured `#e07b39`), then renders a Sources expander. Web and YouTube citations link to their URLs; PDFs show name + page.
- `ChromaStore.list_sources()` now exposes `url` (web/YouTube) and `page_count` (PDF) in each source dict; `source_viewer.py` renders these per source type.
- `MAX_CHUNKS_PER_SOURCE = 3` (in `config/settings.py`) caps chunks per source to ensure diversity; retriever fetches `TOP_K × MAX_CHUNKS_PER_SOURCE` candidates from ChromaDB before applying the cap.
- **YouTube transcript translation (implemented):** when no English transcript exists, the `NoTranscriptFound` fallback now calls `first.translate("en").fetch()` before falling back to the original language. This uses YouTube's own translation endpoint — no API key needed.

**M5 notes:**
- `rag/memory/conversation.py` implements `ConversationMemory`: stores the last `MAX_HISTORY_TURNS` turns as `(user_msg, assistant_msg)` pairs; exposes `get_history()` (flat role/content list), `add_turn()`, `clear()`, and `rewrite_query()`.
- `rewrite_query(query, history)` accepts either external history (from Streamlit session state) or falls back to its own internal turns. Returns the original query unchanged when there is no history to resolve.
- The LLM inside `ConversationMemory` is lazy-initialized (`_get_llm()`) — same pattern as `Generator`.
- **Stateless history management:** history state lives in `st.session_state.messages` (Streamlit layer). `pipeline.ask(query, history=...)` receives it as a parameter. `ConversationMemory` is therefore a stateless rewriting utility inside the pipeline; `add_turn` / `get_history` / `clear` are available but history is not double-stored in the pipeline singleton.
- `pipeline.ask()` rewrites the query only when `history` is non-empty. The **rewritten query** is used for retrieval; the **original query + history** is passed to the generator so the LLM answers in full conversational context.
- `Answer.rewritten_query: str = ""` — non-empty only when rewriting changed the query. Used by the chat UI to show a `🔍 Query rewritten as` expander beneath the assistant's answer.
- `pipeline.clear_history()` delegates to `ConversationMemory.clear()`. Called by the "Clear" button in the chat header.
- Chat UI gains a "Clear" button in the header row (only visible when there are messages).
- Tests: `tests/test_memory.py` (15 unit tests, LLM mocked); `tests/test_pipeline.py` gains 5 M5-specific tests covering rewriting dispatch and `clear_history`.
- `pipeline` exposes three granular public methods used by the chat UI for step-by-step progress: `rewrite_query(query, history)`, `retrieve(query, rerank)` → `list[Document]`, `generate(query, docs, history, rewritten_query)` → `Answer`. `ask()` delegates to these three and remains the high-level entry point for tests and non-UI callers.
- **Live pipeline card:** `app/components/chat.py` uses a single `st.empty()` placeholder (not `st.status()`) for the progress display. Each step calls `.markdown()` on the placeholder immediately, guaranteeing sequential rendering. `st.status()` batches writes on some platforms and was replaced for this reason.
- **Justified answer text:** injected via a one-time `<style>` block targeting `[data-testid="stChatMessage"] p`. Markdown renders normally; CSS aligns the final output.
- **`HF_HUB_DISABLE_SYMLINKS_WARNING=1`** should be set in `.env` on Windows to suppress the HuggingFace symlinks warning (cosmetic only).

**Post-M6 notes:**
- **arXiv ingestion:** `rag/ingestion/arxiv_ingestor.py` parses an arXiv ID from a bare ID (`2305.14283`) or any arXiv URL, fetches the paper title from the arXiv Atom API (`export.arxiv.org/api/query`), downloads the PDF via `requests`, and delegates extraction to `PDFIngestor`. Returned Documents have `source_type="arxiv"` (overridden after PDF extraction via `dataclasses.replace`) and `source_name=<paper title>` (falls back to `"arXiv:<id>"` if the API is unreachable). No API key required. Pipeline routes `source_type="arxiv"` to `ArXivIngestor`. Sidebar has a dedicated 📜 arXiv section with the same counter-key clear pattern as Web/YouTube. The generator's `_format_header()` treats `"arxiv"` identically to `"pdf"` so citation format remains `[PDF: title, page N]`. The source viewer uses the 📜 icon for `source_type="arxiv"`.
- **PDF export:** `app/components/chat.py` exposes a `📥 Export` download button (visible when messages exist) that calls `_export_chat_pdf()`. Uses `fpdf2` (lazy import inside the function). Common non-latin-1 characters (em-dash, smart quotes, ellipsis) are replaced with ASCII equivalents before encoding; long unbreakable tokens are split at 55 chars. **Critical:** all `multi_cell()` calls must pass `new_x=XPos.LMARGIN, new_y=YPos.NEXT` — fpdf2 2.7+ changed the default to `new_x=XPos.RIGHT`, which leaves the cursor at the right edge and causes the next `multi_cell(0, ...)` to compute available width ≈ 0, raising `FPDFException: Not enough horizontal space`. Inline `[N]` citation markers are preserved in the PDF text. Citation locations (URLs) are replaced with `(link)` since PDFs are not interactive.
- **`fpdf2>=2.7`** added to `pyproject.toml` dependencies and `requirements.txt` (regenerated via `uv export`).
- **YouTube ingestion on Streamlit Cloud:** YouTube actively blocks transcript requests from cloud provider IP ranges (AWS/GCP). This is a platform-level restriction — no code fix is possible. The sidebar error message directs users to the **Paste Text** section as a workaround. YouTube ingestion works normally on a locally-hosted instance.
- **Plain text ingestor:** `rag/ingestion/text_ingestor.py` accepts a raw string, returns a single `Document` with `source_type="text"`, `source_id` = first 16 hex chars of SHA-256 of the content. The Chunker splits it downstream. Citation format: `[Text: source name]` (no location field). Regex in `rag/models.py` updated to include "Text". Generator prompt updated. `source_viewer.py` uses "📝" icon. Sidebar section "📝 Paste Text" uses the counter-key clear pattern with both a `text_input` (name) and `text_area` (content).
- **Retrieval parameter sliders:** "Top K chunks" (3–15, default `TOP_K`) and "Max per source" (1–5, default `MAX_CHUNKS_PER_SOURCE`) sliders in the sidebar "⚙️ Retrieval" section. Stored in `st.session_state["top_k"]` and `st.session_state["max_chunks_per_source"]`. `Retriever.retrieve()` and `pipeline.retrieve()` both accept `top_k` and `max_chunks_per_source` keyword args; `_run_pipeline()` in `chat.py` reads them from session state. Changing these at query time does NOT require re-ingestion.
- **Browser timezone detection:** `app/app.py` calls `_detect_timezone()` on every render via `streamlit-javascript`. `st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")` returns `0` on the first render (component initialising) then the IANA timezone string on subsequent renders (auto-triggered rerun). Result cached as `st.session_state.user_tz` (string or `None` for UTC fallback). `source_viewer.py` converts the stored UTC `ingested_at` timestamps using `zoneinfo.ZoneInfo(tz_name)` — showing local time when available, "YYYY-MM-DD HH:MM UTC" as fallback. `tzdata>=2024.1` added to deps for Windows compatibility.
- **Rate limiting:** `MAX_QUESTIONS_PER_SESSION = 10` in `config/settings.py` (0 = unlimited). Tracked via `st.session_state.question_count`. Chat input is disabled and a warning shown when the limit is reached.
- **Hybrid search (BM25 + dense vector):** `rag/retrieval/bm25_index.py` provides a `BM25Index` class (using `BM25Plus` from `rank_bm25`) and a `reciprocal_rank_fusion()` pure function. `BM25Index` is created in `RAGPipeline.__init__()` (per-session — same lifecycle as ChromaStore) and kept in sync: `add()` is called after `ChromaStore.add()` in `ingest()`, `delete()` after `ChromaStore.delete()` in `delete_source()`, `delete_all()` after `ChromaStore.delete_all()`. `Retriever.__init__()` accepts an optional `bm25_index` parameter. When `hybrid=True`, both dense and BM25 results are fused via `reciprocal_rank_fusion()` before cross-encoder re-ranking and per-source cap. Tokenization: `re.findall(r"\w+", text.lower())`. Fusion constant: `RRF_K = 60` in `config/settings.py`. The sidebar has a "Hybrid search (BM25)" toggle (`key="use_hybrid"`). `_run_pipeline()` in `chat.py` reads it and passes `hybrid=` to `pipeline.retrieve()`. The progress card label updates to "Hybrid Retrieve" / "Hybrid Retrieve & Re-rank" when enabled. **Why `BM25Plus` over `BM25Okapi`:** `BM25Okapi` computes negative IDF for terms appearing in all N documents (including N=1), which gives zero or negative scores for matching docs in small corpora. `BM25Plus` uses `idf = log(N+1) - log(df)` which is always positive.

The authoritative milestone plan is `multi-source-rag-project-plan-prompt1.md`.

## Commands

```bash
# Setup (uv manages the venv automatically in .venv/)
cp .env.example .env            # fill in GWDG_API_KEY, GWDG_API_BASE, GWDG_MODEL_NAME
make install                    # uv sync --extra dev

# Run
make run                        # uv run streamlit run app/app.py

# Lint / format (Ruff — no flake8, no black)
make lint                       # uv run ruff check --fix . && uv run ruff format .

# Tests
uv run pytest tests/
uv run pytest tests/test_ingestion.py   # single test file

# Reset the ChromaDB vector store
uv run python scripts/reset_vectorstore.py
```

## Architecture

A conversational RAG assistant that ingests documents from multiple sources, embeds them into a local ChromaDB vector store, and supports multi-turn Q&A with citations via a Streamlit UI.

**Tech stack:** Python 3.10+, LangChain, sentence-transformers (`all-MiniLM-L6-v2`, 384-dim), ChromaDB (ephemeral in-memory per session), any OpenAI-compatible LLM API, PyMuPDF, trafilatura, youtube-transcript-api, fpdf2, streamlit-javascript, tzdata, Streamlit, Ruff.

### Data flow

**Write path (fully built):**
1. User submits a PDF, arXiv ID/URL, web URL, YouTube URL, or pasted text via the Streamlit sidebar.
2. The matching ingestor extracts text (`PDFIngestor`/`ArXivIngestor` → one Document per page; `WebIngestor`/`TextIngestor` → one Document; `YouTubeIngestor` → timestamp-grouped Documents).
3. `Chunker` splits into chunks (`chunk_size=500`, `chunk_overlap=50`) with globally sequential `chunk_index`.
4. `Embedder` produces 384-dim vectors; `ChromaStore` upserts chunks + embeddings into the session's ephemeral in-memory ChromaDB.

**Read path (fully built):**
5. User types a query; if there is prior history, `ConversationMemory.rewrite_query()` reformulates it as a standalone question (one LLM call).
6. `Retriever` embeds the (possibly rewritten) query and fetches the top-k=5 most similar chunks. If re-ranking is enabled, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool before applying the per-source cap.
7. `Generator` calls the GWDG LLM with the **original** query + full history, so the answer is in the conversational context.
8. `Answer.from_raw()` parses citation markers; `app/components/chat.py` renders the answer with coloured numbered citations and a collapsible Sources expander.

### Key abstractions

- `rag/ingestion/base.py` — abstract `Ingestor` base class; all source types subclass this so new sources are drop-in additions.
- `rag/pipeline.py` — orchestrator: wires ingestion → chunking → embedding → ChromaDB (write path) and `ConversationMemory` rewrite → retriever → generator (read path).
- `config/settings.py` — single location for all tunables: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, `TOP_K`, `MAX_HISTORY_TURNS`, `RERANK_MODEL`, `CHROMA_PERSIST_DIR`. Do not scatter these through module code.
- `app/app.py` — Streamlit entry point; UI components under `app/components/`. The `rag/` package is kept framework-agnostic and does not import Streamlit.

### Non-obvious constraints

- **Ingestor base class must exist from Milestone 1**, even before web/YouTube sources are built. The full metadata schema (`source_type`, `source_name`, `source_id`, `page_number`/`url`/`timestamp`, `chunk_index`, `ingested_at`) must also be consistent across all ingestors from day one — the citation rendering contract in the generator prompt depends on it.
- **LLM is called only for answer generation and query rewriting** — not for embedding, chunking, or retrieval. Re-ranking uses a local cross-encoder (no API). Embeddings are cached via Chroma's persistence to conserve the limited GWDG API quota.
- **Citation contract:** the generator prompt instructs the LLM to produce type-prefixed citations: `[PDF: name, page N]` / `[Web: title, URL]` / `[YouTube: title, timestamped_URL]` / `[Text: name]`. The regex in `rag/models.py` matches `(PDF|Web|YouTube|Text)`; `app/components/chat.py` replaces them with numbered `[1]`, `[2]` references. Do not change the format without updating both the generator prompt and the regex.
- **YouTube special case:** transcripts should be chunked by timestamp-boundary segments when possible, not just by character count.
- **Local-first:** no Docker, no external vector DB — ChromaDB is file-persisted, optimized for Streamlit Cloud deployment.