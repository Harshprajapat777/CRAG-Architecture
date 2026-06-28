# CRAG-BOT

A Corrective RAG (CRAG) chatbot grounded in `evangelistsoftware.com` content.

> **Scope of this repo:** **Phase 1 only** — data extraction and Knowledge Base construction.
> The downstream CRAG flow (retrieval, document grading, web-fallback, chat UI) is **not yet implemented**;
> `src/retrieval/` and `src/grading/` are intentional placeholders for Phase 2.

## Tech stack

- Python 3.10+
- LangChain (orchestration; LangGraph planned for the Phase 2 CRAG flow)
- ChromaDB (local persistent vector store)
- OpenAI embeddings (`text-embedding-3-small`, 1536 dims)
- `requests` + `beautifulsoup4` + `trafilatura` (scraping and main-content extraction)
- `pydantic-settings` (centralised config), `tenacity` (retry), `rich` + `tqdm` (UX)
- `pytest` (KB coverage evals)

## Project structure

```
CRAG-BOT/
├── config/
│   └── settings.py            # pydantic-settings; single source of truth, loads .env
├── data/
│   ├── sitemaps/              # sitemap_index.xml + consolidated URL list (xml + txt)
│   ├── raw/                   # raw HTML per URL (gitignored, resumable cache)
│   ├── processed/             # cleaned text + metadata as {slug}.json (gitignored)
│   └── vector_store/          # ChromaDB persistent dir (gitignored)
├── src/
│   ├── ingestion/             # sitemap parser, scraper, scrape pipeline orchestrator
│   ├── embeddings/            # text chunker + OpenAI embedder wrapper
│   ├── vector_store/          # ChromaDB get/upsert/query helpers
│   ├── retrieval/             # placeholder for Phase 2 (CRAG retriever)
│   ├── grading/               # placeholder for Phase 2 (document grader)
│   └── utils/                 # rich logger, IO helpers, URL→slug
├── scripts/
│   ├── 01_extract_sitemap.py  # entry point — Step 1 of the pipeline
│   ├── 02_scrape_pages.py     # entry point — Step 2
│   └── 03_build_vector_store.py  # entry point — Step 3
├── tests/
│   ├── kb_coverage.py         # eval logic + rich CLI report
│   ├── conftest.py            # session-scoped fixture
│   └── test_kb_coverage.py    # pytest threshold assertions
├── .env.example
├── pytest.ini
└── requirements.txt
```

## Data pipeline (Phase 1)

The pipeline is three sequential steps. Each step writes to a dedicated `data/` subdir, so re-runs are resumable (already-processed slugs are skipped) and intermediate state is inspectable.

```
                   ┌───────────────────────────────┐
                   │  evangelistsoftware.com       │
                   │  sitemap_index.xml            │
                   └──────────────┬────────────────┘
                                  │
   ┌──────────────────────────────▼────────────────────────────────┐
   │  STEP 1 — scripts/01_extract_sitemap.py                       │
   │  • Walks sitemap index, fetches every sub-sitemap             │
   │  • Deduplicates by URL                                        │
   │  • Writes data/sitemaps/evangelist_urls.{xml,txt}             │
   └──────────────────────────────┬────────────────────────────────┘
                                  │
   ┌──────────────────────────────▼────────────────────────────────┐
   │  STEP 2 — scripts/02_scrape_pages.py                          │
   │  • requests.get each URL (polite delay, tenacity retries)     │
   │  • Saves raw HTML  → data/raw/{slug}.html                     │
   │  • trafilatura extract (BS4 fallback) → cleaned text          │
   │  • Saves            → data/processed/{slug}.json              │
   │       {url, slug, title, lastmod, text, char_count,           │
   │        scraped_at}                                            │
   │  • Failures aggregated in data/processed/_failures.json       │
   │  • Resumable: existing {slug}.json is skipped                 │
   └──────────────────────────────┬────────────────────────────────┘
                                  │
   ┌──────────────────────────────▼────────────────────────────────┐
   │  STEP 3 — scripts/03_build_vector_store.py                    │
   │  • Loads every processed JSON                                 │
   │  • RecursiveCharacterTextSplitter (size=1000, overlap=150)    │
   │  • Prints token + USD cost estimate BEFORE embedding          │
   │  • OpenAI text-embedding-3-small → 1536-dim vectors           │
   │  • Upserts into ChromaDB with stable IDs                      │
   │    "{source_slug}::{chunk_index}" so re-runs are idempotent   │
   │  • Persists at data/vector_store/                             │
   └───────────────────────────────────────────────────────────────┘
```

At the end of Step 3 the KB is queryable via:

```python
from src.vector_store.chroma_store import get_store, similarity_search
results = similarity_search(get_store(), "your question here", k=4)
```

## Setup

Requires **Python 3.10+** and an **OpenAI API key**.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set a real `OPENAI_API_KEY`. The other defaults (embedding model, chunk size, scraping delay) are sensible — tweak only if needed.

## Run the pipeline

```powershell
python scripts/01_extract_sitemap.py      # ~5 sec   (network only, free)
python scripts/02_scrape_pages.py         # ~3 min   (133 URLs at 1 req/sec)
python scripts/03_build_vector_store.py   # ~30 sec  (uses OpenAI, ~$0.03 one-time)
```

After a clean run you should see ~133 processed JSONs and ~1,400+ chunks in `data/vector_store/`.

## Run the evals

The evals verify that **every sitemap URL → was scraped → ended up in ChromaDB**, with three independent checks (sitemap↔scrape, scrape↔KB, retrievability).

```powershell
# Human-readable report — free, instant, no API calls
python tests/kb_coverage.py

# Same plus a live retrievability probe — uses OpenAI (~$0.001)
python tests/kb_coverage.py --retrieve --sample 15

# Pass/fail assertions on coverage thresholds (CI-style)
pytest                  # 8 structural tests, no API
pytest -m slow          # adds the retrievability assertion (uses OpenAI)
```

A healthy KB reports `Sitemap->Scrape 100% | Scrape->KB 100% | missing-metadata 0` with **VERDICT: OK** at the bottom.

## Status

- **Phase 1** — data extraction + Knowledge Base: **complete and verified** (133 URLs, 1,446 chunks, 100% coverage end-to-end).
- **Phase 2** — CRAG retrieval, document grading, web-fallback search: not started.
- **Phase 3** — chat UI + CAG cache layer: not started.

## Repository

[`Harshprajapat777/CRAG-Architecture`](https://github.com/Harshprajapat777/CRAG-Architecture)
