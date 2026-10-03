from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import httpx
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .config import settings
from .database import db


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_.+-]*", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese: list[str] = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return latin + chinese


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class EmbeddingProvider:
    """OpenAI-compatible embeddings with a deterministic offline fallback."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.embedding_dimension
        self.last_route = "local-hash-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if settings.embedding_api_key:
            try:
                response = httpx.post(
                    f"{settings.embedding_base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
                    json={
                        "model": settings.embedding_model,
                        "input": texts,
                        "dimensions": self.dimension,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
                vectors = [_normalize([float(value) for value in item["embedding"]]) for item in data]
                if len(vectors) == len(texts) and all(len(item) == self.dimension for item in vectors):
                    self.last_route = f"api:{settings.embedding_model}"
                    return vectors
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                pass
        self.last_route = "local-hash-embedding"
        return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        features = tokenize(text)
        normalized = re.sub(r"\s+", "", text.lower())
        features.extend(normalized[index : index + 3] for index in range(max(0, len(normalized) - 2)))
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * (1.0 + min(len(feature), 8) / 8)
        return _normalize(vector)


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    document_name: str
    knowledge_base: str
    content: str
    score: float
    channel: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunk_id,
            "documentId": self.document_id,
            "documentName": self.document_name,
            "knowledgeBase": self.knowledge_base,
            "content": self.content,
            "score": round(self.score, 4),
            "channel": self.channel,
            "scores": {
                "bm25": round(self.bm25_score, 4),
                "vector": round(self.vector_score, 4),
                "rerank": round(self.rerank_score, 4),
            },
        }


class QdrantVectorStore:
    def __init__(
        self,
        embedding: EmbeddingProvider | None = None,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedding = embedding or EmbeddingProvider()
        self.collection_name = collection_name or settings.vector_collection
        self.client = client or QdrantClient(path=str(settings.vector_store_path))
        self._ensure_collection()

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"product-assistant:{chunk_id}"))

    def _ensure_collection(self) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection_name in collections:
            vectors = self.client.get_collection(self.collection_name).config.params.vectors
            existing_size = getattr(vectors, "size", self.embedding.dimension)
            if existing_size != self.embedding.dimension:
                self.client.delete_collection(self.collection_name)
                collections.remove(self.collection_name)
        if self.collection_name not in collections:
            self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, rows: Iterable[dict[str, Any]]) -> None:
        items = list(rows)
        if not items:
            return
        vectors = self.embedding.embed([item["content"] for item in items])
        points = []
        for item, vector in zip(items, vectors, strict=True):
            payload = dict(item)
            points.append(models.PointStruct(id=self.point_id(item["chunk_id"]), vector=vector, payload=payload))
        self.client.upsert(self.collection_name, points=points, wait=True)

    def delete(self, chunk_ids: Iterable[str]) -> None:
        ids = [self.point_id(chunk_id) for chunk_id in chunk_ids]
        if ids:
            self.client.delete(self.collection_name, points_selector=ids, wait=True)

    def synchronize(self, rows: list[dict[str, Any]]) -> None:
        expected = {row["chunk_id"]: row for row in rows}
        existing: dict[str, dict[str, Any]] = {}
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                if payload.get("chunk_id"):
                    existing[payload["chunk_id"]] = payload
            if offset is None:
                break
        stale = set(existing) - set(expected)
        changed = [
            row for chunk_id, row in expected.items()
            if chunk_id not in existing or existing[chunk_id].get("content_hash") != row.get("content_hash")
        ]
        self.delete(stale)
        self.upsert(changed)

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = self.embedding.embed([query])[0]
        response = self.client.query_points(
            self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            (str(point.payload.get("chunk_id")), float(point.score))
            for point in response.points
            if point.payload and point.payload.get("chunk_id")
        ]


class Reranker:
    """Dedicated rerank API with an explainable local scoring fallback."""

    def __init__(self, tokenizer: Callable[[str], list[str]] = tokenize) -> None:
        self.tokenizer = tokenizer
        self.last_route = "local-feature-reranker"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        if settings.rerank_api_key:
            try:
                response = httpx.post(
                    settings.rerank_url,
                    headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
                    json={
                        "model": settings.rerank_model,
                        "query": query,
                        "documents": [hit.content for hit in hits],
                        "top_n": min(top_k, len(hits)),
                    },
                    timeout=60,
                )
                response.raise_for_status()
                ranked: list[SearchHit] = []
                for item in response.json()["results"]:
                    hit = hits[int(item["index"])]
                    hit.rerank_score = float(item["relevance_score"])
                    hit.score = hit.rerank_score
                    ranked.append(hit)
                self.last_route = f"api:{settings.rerank_model}"
                return ranked[:top_k]
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                pass
        query_tokens = set(self.tokenizer(query))
        normalized_query = re.sub(r"\s+", "", query.lower())
        for hit in hits:
            content_tokens = set(self.tokenizer(hit.content))
            coverage = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            phrase_bonus = 1.0 if normalized_query and normalized_query in re.sub(r"\s+", "", hit.content.lower()) else 0.0
            hit.rerank_score = 0.55 * coverage + 0.25 * phrase_bonus + 0.20 * hit.score
            hit.score = hit.rerank_score
        self.last_route = "local-feature-reranker"
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


def bm25_rank(query: str, corpus: list[str]) -> list[tuple[int, float]]:
    if not corpus:
        return []
    model = BM25Okapi([tokenize(text) for text in corpus])
    scores = model.get_scores(tokenize(query))
    return sorted(
        [(index, float(score)) for index, score in enumerate(scores)],
        key=lambda item: item[1],
        reverse=True,
    )


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.reranker = reranker or Reranker()

    def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        rewritten = question.strip()
        with db.connect() as connection:
            mappings = connection.execute(
                "SELECT source_term, target_term FROM mappings WHERE enabled=1 ORDER BY priority DESC"
            ).fetchall()
        for row in mappings:
            if row["source_term"] in rewritten:
                rewritten += f" {row['target_term']}"
        if history and len(rewritten) < 18 and re.search(r"(它|这个|那|呢|如何|区别)", rewritten):
            previous = next((item["content"] for item in reversed(history) if item["role"] == "user"), "")
            if previous:
                rewritten = f"基于上一问“{previous[:100]}”，{rewritten}"
        return rewritten

    @staticmethod
    def _rows() -> list[dict[str, Any]]:
        with db.connect() as connection:
            rows = connection.execute(
                """SELECT c.id chunk_id, c.doc_id document_id, c.content, c.content_hash,
                          d.doc_name document_name, kb.name knowledge_base
                   FROM chunks c JOIN documents d ON d.id=c.doc_id
                   JOIN knowledge_bases kb ON kb.id=c.kb_id
                   WHERE c.enabled=1 AND d.enabled=1 AND d.status='ready'"""
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, intent_code: str | None = None, top_k: int | None = None) -> list[SearchHit]:
        rows = self._rows()
        if not rows or not tokenize(query):
            return []
        self.vector_store.synchronize(rows)
        candidate_limit = min(len(rows), max((top_k or settings.retrieval_top_k) * 4, 12))

        bm25_ranked_pairs = bm25_rank(query, [row["content"] for row in rows])[:candidate_limit]
        vector_ranked = self.vector_store.search(query, candidate_limit)

        by_id = {row["chunk_id"]: row for row in rows}
        bm25_ids = [rows[index]["chunk_id"] for index, _ in bm25_ranked_pairs]
        vector_ids = [chunk_id for chunk_id, _ in vector_ranked if chunk_id in by_id]
        rrf_scores = reciprocal_rank_fusion([bm25_ids, vector_ids], settings.rrf_k)
        fused: dict[str, dict[str, float]] = {
            chunk_id: {"rrf": score, "bm25": 0.0, "vector": 0.0}
            for chunk_id, score in rrf_scores.items()
        }
        for index, score in bm25_ranked_pairs:
            chunk_id = rows[index]["chunk_id"]
            fused[chunk_id]["bm25"] = score
        for chunk_id, score in vector_ranked:
            if chunk_id not in by_id:
                continue
            fused[chunk_id]["vector"] = score

        candidates = []
        for chunk_id, values in sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)[:candidate_limit]:
            row = by_id[chunk_id]
            candidates.append(SearchHit(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                document_name=row["document_name"],
                knowledge_base=row["knowledge_base"],
                content=row["content"],
                score=values["rrf"],
                channel="bm25+qdrant+rrf+rerank" if not intent_code else "intent+bm25+qdrant+rrf+rerank",
                bm25_score=values["bm25"],
                vector_score=values["vector"],
            ))
        return self.reranker.rerank(query, candidates, top_k or settings.retrieval_top_k)


_vector_store: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store


def close_vector_store() -> None:
    global _vector_store
    if _vector_store is not None:
        _vector_store.client.close()
        _vector_store = None


def index_chunks(rows: Iterable[dict[str, Any]]) -> None:
    get_vector_store().upsert(rows)


def remove_chunks(chunk_ids: Iterable[str]) -> None:
    get_vector_store().delete(chunk_ids)
