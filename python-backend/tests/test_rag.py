from pathlib import Path

from app.database import Database
from app.rag import HybridRetriever, structure_chunks, tokenize


def test_tokenize_supports_chinese_and_english() -> None:
    tokens = tokenize("RAG 产品价格")
    assert "rag" in tokens
    assert "产品" in tokens
    assert "价格" in tokens


def test_structure_chunks_preserves_headings() -> None:
    text = "# 产品\n\n产品介绍。\n\n## 价格\n\n价格政策。"
    chunks = structure_chunks(text, max_chars=30, overlap=0)
    assert any("# 产品" in chunk for chunk in chunks)
    assert any("## 价格" in chunk for chunk in chunks)


def test_database_initializes(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.db")
    database.initialize()
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1

