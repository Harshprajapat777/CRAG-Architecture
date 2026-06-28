# CRAG-BOT

CRAG-BOT — a Corrective RAG chatbot grounded in evangelistsoftware.com content.

## Tech stack

- Python 3.10+
- LangChain (orchestration)
- ChromaDB (vector store, local persistence)
- OpenAI embeddings (`text-embedding-3-small`)
- `requests`, `beautifulsoup4`, `trafilatura` (scraping and content extraction)

## Folder structure

```
CRAG-BOT/
├── config/
├── data/
│   ├── sitemaps/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
├── src/
│   ├── ingestion/
│   ├── embeddings/
│   ├── vector_store/
│   ├── retrieval/
│   ├── grading/
│   └── utils/
├── scripts/
└── tests/
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` to add your `OPENAI_API_KEY`.

## Pipeline

```
python scripts/01_extract_sitemap.py
python scripts/02_scrape_pages.py
python scripts/03_build_vector_store.py
```

## Status

Step 1 (data extraction + vector store) — in progress.
