from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

try:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI 产品智能助手"
    api_prefix: str = "/api/product-assistant"
    database_path: Path = Path(os.getenv("DATABASE_PATH", BACKEND_DIR / "data" / "assistant.db"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", BACKEND_DIR / "data" / "uploads"))
    seed_docs_dir: Path = Path(
        os.getenv("SEED_DOCS_DIR", PROJECT_DIR / "resources" / "docs" / "knowledge" / "product")
    )
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    secondary_api_key: str = os.getenv("SECONDARY_LLM_API_KEY", "")
    secondary_base_url: str = os.getenv("SECONDARY_LLM_BASE_URL", "")
    secondary_model: str = os.getenv("SECONDARY_LLM_MODEL", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    vector_store_path: Path = Path(os.getenv("VECTOR_STORE_PATH", BACKEND_DIR / "data" / "qdrant"))
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "product_knowledge")
    rerank_api_key: str = os.getenv("RERANK_API_KEY", "")
    rerank_url: str = os.getenv("RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    query_rewrite_enabled: bool = _env_bool("QUERY_REWRITE_ENABLED", True)
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "10000"))
    seed_demo_data: bool = _env_bool("SEED_DEMO_DATA", True)


settings = Settings()

