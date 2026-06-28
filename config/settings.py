from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM / embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Vector store
    chroma_persist_dir: str = "./data/vector_store"
    chroma_collection_name: str = "evangelist_kb"

    # Scraping
    scrape_user_agent: str = "CRAG-BOT/1.0 (+https://evangelistsoftware.com)"
    scrape_request_delay_sec: float = 1.0
    scrape_max_retries: int = 3
    scrape_timeout_sec: int = 30

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Source
    sitemap_index_url: str = "https://evangelistsoftware.com/sitemap_index.xml"

    # Derived paths
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def sitemaps_dir(self) -> Path:
        return self.data_dir / "sitemaps"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def vector_store_dir(self) -> Path:
        path = Path(self.chroma_persist_dir)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


settings = Settings()
