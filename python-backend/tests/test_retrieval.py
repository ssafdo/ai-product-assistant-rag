from types import SimpleNamespace

from qdrant_client import QdrantClient

import app.retrieval as retrieval_module
from app.retrieval import (
    EmbeddingProvider,
    QdrantVectorStore,
    Reranker,
    SearchHit,
    bm25_rank,
    reciprocal_rank_fusion,
)


class FakeEmbedding(EmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimension=3)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] if "价格" in text else [0.0, 1.0, 0.0] for text in texts]


def test_bm25_ranks_matching_product_content_first() -> None:
    corpus = [
        "企业版按坐席数量计费，支持私有化部署",
        "控制台支持修改用户头像",
        "工作台可以导出通话记录",
    ]
    ranked = bm25_rank("企业版价格如何计费", corpus)
    assert ranked[0][0] == 0
    assert ranked[0][1] > ranked[1][1]


def test_qdrant_vector_store_searches_embeddings() -> None:
    store = QdrantVectorStore(
        embedding=FakeEmbedding(),
        client=QdrantClient(location=":memory:"),
        collection_name="test_vectors",
    )
    store.upsert([
        {"chunk_id": "price", "content": "产品价格", "content_hash": "a"},
        {"chunk_id": "profile", "content": "修改头像", "content_hash": "b"},
    ])
    assert store.search("价格", 1)[0][0] == "price"


def test_embedding_provider_calls_compatible_api(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [3.0, 4.0, 0.0]}]}

    monkeypatch.setattr(retrieval_module, "settings", SimpleNamespace(
        embedding_api_key="key",
        embedding_base_url="https://embedding.example/v1",
        embedding_model="embedding-model",
    ))
    monkeypatch.setattr(retrieval_module.httpx, "post", lambda *args, **kwargs: Response())
    provider = EmbeddingProvider(dimension=3)
    assert provider.embed(["test"])[0] == [0.6, 0.8, 0.0]
    assert provider.last_route == "api:embedding-model"


def test_rrf_combines_sparse_and_dense_rankings() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_local_reranker_promotes_exact_phrase() -> None:
    hits = [
        SearchHit("1", "d1", "a.md", "kb", "企业版支持部署", 0.03, "hybrid"),
        SearchHit("2", "d2", "b.md", "kb", "企业版价格按坐席计费", 0.02, "hybrid"),
    ]
    ranked = Reranker().rerank("企业版价格", hits, 2)
    assert ranked[0].chunk_id == "2"
    assert ranked[0].rerank_score > ranked[1].rerank_score


def test_reranker_calls_dedicated_model_api(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"results": [{"index": 1, "relevance_score": 0.91}]}

    monkeypatch.setattr(retrieval_module, "settings", SimpleNamespace(
        rerank_api_key="key",
        rerank_url="https://rerank.example/v1/rerank",
        rerank_model="rerank-model",
    ))
    monkeypatch.setattr(retrieval_module.httpx, "post", lambda *args, **kwargs: Response())
    hits = [
        SearchHit("1", "d1", "a.md", "kb", "first", 0.03, "hybrid"),
        SearchHit("2", "d2", "b.md", "kb", "second", 0.02, "hybrid"),
    ]
    ranked = Reranker().rerank("query", hits, 1)
    assert ranked[0].chunk_id == "2"
    assert ranked[0].rerank_score == 0.91
