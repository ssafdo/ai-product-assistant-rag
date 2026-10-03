from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import settings


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 180_000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        _, salt, expected = encoded.split("$", 2)
    except ValueError:
        return False
    actual = hash_password(password, salt).rsplit("$", 1)[-1]
    return hmac.compare_digest(actual, expected)


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'user', avatar TEXT, create_time TEXT NOT NULL, update_time TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
  token TEXT PRIMARY KEY, user_id TEXT NOT NULL, create_time TEXT NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS knowledge_bases (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, embedding_model TEXT NOT NULL,
  collection_name TEXT NOT NULL, created_by TEXT, create_time TEXT NOT NULL, update_time TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, doc_name TEXT NOT NULL, source_type TEXT,
  source_location TEXT, schedule_enabled INTEGER DEFAULT 0, schedule_cron TEXT,
  enabled INTEGER DEFAULT 1, file_url TEXT, file_type TEXT, file_size INTEGER,
  process_mode TEXT DEFAULT 'chunk', chunk_strategy TEXT DEFAULT 'structure', chunk_config TEXT,
  pipeline_id TEXT, status TEXT DEFAULT 'ready', raw_text TEXT, create_time TEXT NOT NULL, update_time TEXT NOT NULL,
  FOREIGN KEY(kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL, content_hash TEXT NOT NULL, char_count INTEGER NOT NULL,
  token_count INTEGER NOT NULL, enabled INTEGER DEFAULT 1, create_time TEXT NOT NULL, update_time TEXT NOT NULL,
  FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chunk_logs (
  id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, status TEXT NOT NULL, process_mode TEXT,
  chunk_strategy TEXT, pipeline_id TEXT, pipeline_name TEXT, extract_duration INTEGER,
  chunk_duration INTEGER, embed_duration INTEGER, persist_duration INTEGER, other_duration INTEGER,
  total_duration INTEGER, chunk_count INTEGER, error_message TEXT, start_time TEXT, end_time TEXT, create_time TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL, create_time TEXT NOT NULL, update_time TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, vote INTEGER, create_time TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sample_questions (
  id TEXT PRIMARY KEY, title TEXT, description TEXT, question TEXT NOT NULL, create_time TEXT NOT NULL, update_time TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mappings (
  id TEXT PRIMARY KEY, source_term TEXT NOT NULL, target_term TEXT NOT NULL, match_type INTEGER DEFAULT 1,
  priority INTEGER DEFAULT 0, enabled INTEGER DEFAULT 1, remark TEXT, create_time TEXT NOT NULL, update_time TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS intents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kb_id TEXT, intent_code TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  level INTEGER DEFAULT 1, parent_code TEXT, description TEXT, examples TEXT, collection_name TEXT,
  mcp_tool_id TEXT, top_k INTEGER DEFAULT 5, kind INTEGER DEFAULT 0, sort_order INTEGER DEFAULT 0,
  enabled INTEGER DEFAULT 1, prompt_snippet TEXT, prompt_template TEXT, param_prompt_template TEXT
);
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY, trace_name TEXT, entry_method TEXT, conversation_id TEXT, task_id TEXT,
  user_id TEXT, username TEXT, status TEXT, error_message TEXT, duration_ms INTEGER,
  start_time TEXT, end_time TEXT
);
CREATE TABLE IF NOT EXISTS trace_nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT NOT NULL, node_id TEXT NOT NULL, parent_node_id TEXT,
  depth INTEGER, node_type TEXT, node_name TEXT, class_name TEXT, method_name TEXT, status TEXT,
  error_message TEXT, duration_ms INTEGER, start_time TEXT, end_time TEXT, input_json TEXT, output_json TEXT
);
CREATE TABLE IF NOT EXISTS pipelines (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, created_by TEXT, nodes_json TEXT,
  create_time TEXT NOT NULL, update_time TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingestion_tasks (
  id TEXT PRIMARY KEY, pipeline_id TEXT NOT NULL, source_type TEXT, source_location TEXT, source_file_name TEXT,
  status TEXT, chunk_count INTEGER, error_message TEXT, logs_json TEXT, metadata_json TEXT,
  started_at TEXT, completed_at TEXT, created_by TEXT, create_time TEXT, update_time TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kb ON chunks(kb_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_traces_start ON traces(start_time DESC);
"""


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            timestamp = now_iso()
            connection.execute(
                "INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                (new_id(), "admin", hash_password("admin"), "admin", None, timestamp, timestamp),
            )
            connection.execute(
                "INSERT OR IGNORE INTO sample_questions VALUES (?, ?, ?, ?, ?, ?)",
                (new_id(), "产品能力", "了解产品定位和核心能力", "这个 AI 产品智能助手能解决哪些问题？", timestamp, timestamp),
            )
            defaults = [
                ("报价", "套餐 价格 政策"),
                ("版本权益", "套餐 功能 权益"),
                ("怎么部署", "交付 部署 环境"),
            ]
            for source, target in defaults:
                connection.execute(
                    "INSERT OR IGNORE INTO mappings(id, source_term, target_term, remark, create_time, update_time) VALUES (?, ?, ?, ?, ?, ?)",
                    (new_id(), source, target, "内置产品术语归一化", timestamp, timestamp),
                )
            intents = [
                ("product_overview", "产品介绍", "功能 能力 场景 产品", "回答时说明产品定位、目标用户与能力边界"),
                ("pricing", "套餐与价格", "价格 报价 套餐 权益 版本", "具体报价必须提示以商务确认为准"),
                ("delivery", "交付与部署", "部署 交付 验收 环境 私有化", "优先给出准备项和验收标准"),
                ("sales", "售前支持", "售前 客户 话术 竞品 方案", "使用客观、可核验的产品口径"),
                ("release", "版本发布", "版本 发布 更新 升级", "区分已发布能力与规划"),
            ]
            for code, name, examples, prompt in intents:
                connection.execute(
                    "INSERT OR IGNORE INTO intents(intent_code, name, examples, prompt_snippet) VALUES (?, ?, ?, ?)",
                    (code, name, examples, prompt),
                )
            pipeline_id = "default-product-docs"
            nodes = [
                {"nodeId": "parse", "nodeType": "parser", "settings": {}},
                {"nodeId": "chunk", "nodeType": "chunker", "settings": {"strategy": "structure"}},
                {"nodeId": "index", "nodeType": "indexer", "settings": {"engine": "sqlite-hybrid"}},
            ]
            connection.execute(
                "INSERT OR IGNORE INTO pipelines VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pipeline_id, "默认产品文档入库", "解析、结构化切分并建立混合检索索引", "system", dumps(nodes), timestamp, timestamp),
            )


db = Database()
