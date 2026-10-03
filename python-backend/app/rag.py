from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import settings
from .database import db, dumps, loads, new_id, now_iso


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    latin = re.findall(r"[a-z0-9][a-z0-9_.+-]*", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese: list[str] = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return latin + chinese


def structure_chunks(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    sections = re.split(r"(?=^#{1,4}\s+)", text, flags=re.MULTILINE)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()]
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = f"{current[-overlap:]}\n\n{paragraph}" if overlap else paragraph
                while len(current) > max_chars:
                    chunks.append(current[:max_chars])
                    current = current[max_chars - overlap :]
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def ingest_text(
    kb_id: str,
    doc_name: str,
    text: str,
    *,
    source_type: str = "file",
    source_location: str | None = None,
    file_size: int | None = None,
    chunk_strategy: str = "structure",
    document_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    timestamp = now_iso()
    document_id = document_id or new_id()
    chunks = structure_chunks(text) if chunk_strategy == "structure" else structure_chunks(text, 700, 100)
    with db.connect() as connection:
        existing = connection.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if existing:
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (document_id,))
            connection.execute(
                "UPDATE documents SET raw_text=?, status='processing', update_time=? WHERE id=?",
                (text, timestamp, document_id),
            )
        else:
            connection.execute(
                """INSERT INTO documents(
                    id, kb_id, doc_name, source_type, source_location, enabled, file_type, file_size,
                    process_mode, chunk_strategy, status, raw_text, create_time, update_time
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, 'chunk', ?, 'processing', ?, ?, ?)""",
                (
                    document_id,
                    kb_id,
                    doc_name,
                    source_type,
                    source_location,
                    Path(doc_name).suffix.lower().lstrip("."),
                    file_size if file_size is not None else len(text.encode("utf-8")),
                    chunk_strategy,
                    text,
                    timestamp,
                    timestamp,
                ),
            )
        for index, content in enumerate(chunks):
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            connection.execute(
                """INSERT INTO chunks(
                    id, kb_id, doc_id, chunk_index, content, content_hash, char_count,
                    token_count, enabled, create_time, update_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    new_id(), kb_id, document_id, index, content, content_hash,
                    len(content), len(tokenize(content)), timestamp, timestamp,
                ),
            )
        connection.execute(
            "UPDATE documents SET status='ready', update_time=? WHERE id=?",
            (timestamp, document_id),
        )
        duration = int((time.perf_counter() - started) * 1000)
        connection.execute(
            """INSERT INTO chunk_logs(
                id, doc_id, status, process_mode, chunk_strategy, total_duration, chunk_count,
                start_time, end_time, create_time
            ) VALUES (?, ?, 'success', 'chunk', ?, ?, ?, ?, ?, ?)""",
            (new_id(), document_id, chunk_strategy, duration, len(chunks), timestamp, now_iso(), timestamp),
        )
    return get_document(document_id)


def get_document(document_id: str) -> dict[str, Any]:
    with db.connect() as connection:
        row = connection.execute(
            """SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id) chunk_count
               FROM documents d WHERE d.id=?""",
            (document_id,),
        ).fetchone()
    if not row:
        raise KeyError("文档不存在")
    return document_to_api(dict(row))


def document_to_api(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"], "kbId": row["kb_id"], "docName": row["doc_name"],
        "sourceType": row.get("source_type"), "sourceLocation": row.get("source_location"),
        "scheduleEnabled": row.get("schedule_enabled", 0), "scheduleCron": row.get("schedule_cron"),
        "enabled": bool(row.get("enabled", 1)), "chunkCount": row.get("chunk_count", 0),
        "fileUrl": row.get("file_url"), "fileType": row.get("file_type"), "fileSize": row.get("file_size"),
        "processMode": row.get("process_mode"), "chunkStrategy": row.get("chunk_strategy"),
        "chunkConfig": row.get("chunk_config"), "pipelineId": row.get("pipeline_id"),
        "status": row.get("status"), "createTime": row.get("create_time"), "updateTime": row.get("update_time"),
    }


def seed_product_knowledge() -> None:
    if not settings.seed_demo_data or not settings.seed_docs_dir.exists():
        return
    with db.connect() as connection:
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if count:
            return
        timestamp = now_iso()
        kb_id = "product-knowledge"
        connection.execute(
            "INSERT OR IGNORE INTO knowledge_bases VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kb_id, "产品知识库", "local-hybrid", "product_assistant_store", "system", timestamp, timestamp),
        )
    for path in sorted(settings.seed_docs_dir.rglob("*.md")):
        ingest_text(
            kb_id,
            path.name,
            path.read_text(encoding="utf-8"),
            source_type="seed",
            source_location=str(path.relative_to(settings.seed_docs_dir)),
            file_size=path.stat().st_size,
        )


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    document_name: str
    knowledge_base: str
    content: str
    score: float
    channel: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunkId": self.chunk_id,
            "documentId": self.document_id,
            "documentName": self.document_name,
            "knowledgeBase": self.knowledge_base,
            "content": self.content,
            "score": round(self.score, 4),
            "channel": self.channel,
        }


class HybridRetriever:
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

    def search(self, query: str, intent_code: str | None = None, top_k: int | None = None) -> list[SearchHit]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        with db.connect() as connection:
            rows = connection.execute(
                """SELECT c.id chunk_id, c.doc_id, c.content, d.doc_name, kb.name kb_name
                   FROM chunks c JOIN documents d ON d.id=c.doc_id
                   JOIN knowledge_bases kb ON kb.id=c.kb_id
                   WHERE c.enabled=1 AND d.enabled=1 AND d.status='ready'"""
            ).fetchall()
        hits: list[SearchHit] = []
        intent_terms = set(tokenize(intent_code or ""))
        for row in rows:
            content_tokens = set(tokenize(row["content"]))
            overlap = query_tokens & content_tokens
            if not overlap:
                continue
            lexical = sum(2.0 if len(token) > 1 else 0.35 for token in overlap)
            coverage = len(overlap) / max(1, len(query_tokens))
            exact_bonus = 2.0 if query.lower() in row["content"].lower() else 0.0
            intent_bonus = 0.6 * len(intent_terms & content_tokens)
            length_penalty = math.log(max(20, len(content_tokens)), 10)
            score = (lexical + 5 * coverage + exact_bonus + intent_bonus) / length_penalty
            hits.append(
                SearchHit(
                    row["chunk_id"], row["doc_id"], row["doc_name"], row["kb_name"],
                    row["content"], score, "hybrid-global" if not intent_code else "intent-directed",
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: top_k or settings.retrieval_top_k]


class IntentRouter:
    def route(self, question: str) -> dict[str, Any]:
        query_tokens = set(tokenize(question))
        best = {"code": "general", "name": "通用产品问答", "score": 0.0, "prompt": ""}
        with db.connect() as connection:
            rows = connection.execute("SELECT * FROM intents WHERE enabled=1").fetchall()
        for row in rows:
            example_tokens = set(tokenize(row["examples"] or ""))
            common = query_tokens & example_tokens
            score = sum(2 if len(token) > 1 else 0.25 for token in common)
            if score > best["score"]:
                best = {
                    "code": row["intent_code"], "name": row["name"], "score": score,
                    "prompt": row["prompt_snippet"] or "", "topK": row["top_k"] or settings.retrieval_top_k,
                }
        return best


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[str, Callable[..., Any]]] = {}

    def register(self, name: str, description: str, function: Callable[..., Any]) -> None:
        self._tools[name] = (description, function)

    def execute(self, name: str, **arguments: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"工具 {name} 不存在")
        return self._tools[name][1](**arguments)

    def descriptions(self) -> list[dict[str, str]]:
        return [{"name": name, "description": item[0]} for name, item in self._tools.items()]


class ContextBuilder:
    def build(self, question: str, history: list[dict[str, str]], hits: list[SearchHit]) -> str:
        # GSSC: gather is done by retrieval/history, then select, structure and compress here.
        selected: list[str] = []
        used = 0
        for index, hit in enumerate(hits, start=1):
            block = f"[资料{index}] {hit.document_name}\n{hit.content.strip()}"
            if used + len(block) > settings.max_context_chars:
                remaining = settings.max_context_chars - used
                if remaining > 200:
                    selected.append(block[:remaining])
                break
            selected.append(block)
            used += len(block)
        recent = history[-8:]
        history_text = "\n".join(f"{item['role']}: {item['content'][:600]}" for item in recent)
        return (
            "[Instruction]\n你是严谨的 AI 产品智能助手。只依据证据回答；资料不足时明确说明，"
            "不得编造价格、合同、客户隐私或未发布路线图。回答中用 [资料N] 标注依据。\n\n"
            f"[Question]\n{question}\n\n[Evidence]\n" + ("\n\n".join(selected) or "未检索到相关资料")
            + f"\n\n[Context]\n{history_text or '无历史对话'}"
        )


class ModelRouter:
    async def complete(self, context: str, intent: dict[str, Any], hits: list[SearchHit]) -> tuple[str, str]:
        providers = []
        if settings.llm_api_key:
            providers.append((settings.llm_base_url, settings.llm_api_key, settings.llm_model, "primary"))
        if settings.secondary_api_key and settings.secondary_base_url and settings.secondary_model:
            providers.append(
                (settings.secondary_base_url, settings.secondary_api_key, settings.secondary_model, "secondary")
            )
        system = "你是企业 AI 产品智能助手。回答准确、简洁、可执行，并保留证据编号。"
        if intent.get("prompt"):
            system += "\n意图约束：" + intent["prompt"]
        for base_url, api_key, model, route_name in providers:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": context}],
                            "temperature": 0.2,
                        },
                    )
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"].strip()
                    if answer:
                        return answer, f"{route_name}:{model}"
            except (httpx.HTTPError, KeyError, IndexError, TypeError):
                continue
        return self._grounded_fallback(hits), "local-grounded-fallback"

    @staticmethod
    def _grounded_fallback(hits: list[SearchHit]) -> str:
        if not hits:
            return (
                "当前知识库没有检索到足够信息，因此我不能给出可靠结论。"
                "建议补充相关产品手册、FAQ 或最新业务口径后再查询。"
            )
        paragraphs = []
        for index, hit in enumerate(hits[:3], start=1):
            body = re.sub(r"^#{1,4}\s+", "", hit.content.strip(), flags=re.MULTILINE)
            body = body[:520].rstrip()
            paragraphs.append(f"{body} [资料{index}]")
        sources = "\n".join(f"- [资料{i}] {hit.document_name}" for i, hit in enumerate(hits[:3], start=1))
        return "根据当前产品资料：\n\n" + "\n\n".join(paragraphs) + "\n\n参考资料：\n" + sources


@dataclass
class AgentResult:
    answer: str
    rewritten_question: str
    intent: dict[str, Any]
    hits: list[SearchHit]
    model_route: str
    trace_id: str
    task_id: str
    thinking: str = ""


class TraceRecorder:
    def __init__(self, conversation_id: str, task_id: str, user: dict[str, Any]) -> None:
        self.trace_id = new_id()
        self.conversation_id = conversation_id
        self.task_id = task_id
        self.user = user
        self.started = time.perf_counter()
        self.start_time = now_iso()
        with db.connect() as connection:
            connection.execute(
                "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.trace_id, "Agentic RAG 产品问答", "rag_chat", conversation_id, task_id,
                    user["id"], user["username"], "running", None, None, self.start_time, None,
                ),
            )

    def node(self, name: str, node_type: str, started: float, output: Any, input_value: Any = None) -> None:
        timestamp = now_iso()
        with db.connect() as connection:
            connection.execute(
                """INSERT INTO trace_nodes(
                    trace_id, node_id, depth, node_type, node_name, class_name, method_name, status,
                    duration_ms, start_time, end_time, input_json, output_json
                ) VALUES (?, ?, 1, ?, ?, 'ProductAssistantAgent', ?, 'success', ?, ?, ?, ?, ?)""",
                (
                    self.trace_id, new_id(), node_type, name, name,
                    int((time.perf_counter() - started) * 1000), timestamp, now_iso(),
                    dumps(input_value) if input_value is not None else None,
                    dumps(output) if output is not None else None,
                ),
            )

    def finish(self, status: str = "success", error: str | None = None) -> None:
        with db.connect() as connection:
            connection.execute(
                "UPDATE traces SET status=?, error_message=?, duration_ms=?, end_time=? WHERE trace_id=?",
                (status, error, int((time.perf_counter() - self.started) * 1000), now_iso(), self.trace_id),
            )


class ProductAssistantAgent:
    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.intent_router = IntentRouter()
        self.context_builder = ContextBuilder()
        self.model_router = ModelRouter()
        self.tools = ToolRegistry()
        self.tools.register("search_product_knowledge", "检索产品知识库并返回证据", self.retriever.search)
        self.tools.register("get_ingestion_task", "按任务 ID 查询文档入库状态", self._get_ingestion_task)

    @staticmethod
    def _get_ingestion_task(task_id: str) -> dict[str, Any]:
        with db.connect() as connection:
            row = connection.execute("SELECT * FROM ingestion_tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else {"error": "未找到入库任务"}

    async def run(
        self,
        question: str,
        conversation_id: str,
        task_id: str,
        user: dict[str, Any],
        history: list[dict[str, str]],
        deep_thinking: bool = False,
    ) -> AgentResult:
        trace = TraceRecorder(conversation_id, task_id, user)
        try:
            started = time.perf_counter()
            rewritten = self.retriever.rewrite(question, history) if settings.query_rewrite_enabled else question
            trace.node("问题改写", "rewrite", started, {"rewrittenQuestion": rewritten}, {"question": question})

            started = time.perf_counter()
            intent = self.intent_router.route(rewritten)
            trace.node("意图路由", "intent", started, intent, {"question": rewritten})

            started = time.perf_counter()
            hits = self.tools.execute(
                "search_product_knowledge",
                query=rewritten,
                intent_code=intent["code"],
                top_k=intent.get("topK", settings.retrieval_top_k),
            )
            trace.node("知识检索工具", "tool", started, [hit.as_dict() for hit in hits], {"tool": "search_product_knowledge"})

            started = time.perf_counter()
            context = self.context_builder.build(question, history, hits)
            trace.node("上下文构建", "context", started, {"chars": len(context), "evidenceCount": len(hits)})

            started = time.perf_counter()
            answer, route = await self.model_router.complete(context, intent, hits)
            trace.node("模型路由与生成", "generation", started, {"route": route, "answerChars": len(answer)})
            thinking = ""
            if deep_thinking:
                thinking = (
                    f"已将问题改写为“{rewritten}”，识别为“{intent['name']}”意图，"
                    f"通过知识检索工具筛选出 {len(hits)} 条证据，并按相关性压缩上下文后生成回答。"
                )
            trace.finish()
            return AgentResult(answer, rewritten, intent, hits, route, trace.trace_id, task_id, thinking)
        except Exception as exc:
            trace.finish("error", str(exc)[:1000])
            raise


agent = ProductAssistantAgent()
