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
    query_rewrite_enabled: bool = _env_bool("QUERY_REWRITE_ENABLED", True)
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "10000"))
    seed_demo_data: bool = _env_bool("SEED_DEMO_DATA", True)


settings = Settings()

