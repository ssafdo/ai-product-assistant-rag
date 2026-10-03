from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .config import settings
from .context_engineering import ContextAssembler
from .database import db, dumps, new_id, now_iso
from .react_agent import OpenAIToolCallingModel, ReActEngine, ToolRegistry
from .retrieval import HybridRetriever, SearchHit, index_chunks, remove_chunks, tokenize


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
    indexed_rows: list[dict[str, Any]] = []
    removed_chunk_ids: list[str] = []
    with db.connect() as connection:
        existing = connection.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if existing:
            removed_chunk_ids = [
                row["id"] for row in connection.execute("SELECT id FROM chunks WHERE doc_id = ?", (document_id,))
            ]
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
            chunk_id = new_id()
            connection.execute(
                """INSERT INTO chunks(
                    id, kb_id, doc_id, chunk_index, content, content_hash, char_count,
                    token_count, enabled, create_time, update_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    chunk_id, kb_id, document_id, index, content, content_hash,
                    len(content), len(tokenize(content)), timestamp, timestamp,
                ),
            )
            indexed_rows.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": doc_name,
                "knowledge_base": kb_id,
                "content": content,
                "content_hash": content_hash,
            })
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
    remove_chunks(removed_chunk_ids)
    index_chunks(indexed_rows)
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


class ModelRouter:
    def __init__(self, client_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        self.client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=60))

    async def stream_complete(
        self,
        context: str,
        intent: dict[str, Any],
        hits: list[SearchHit],
        on_token: Callable[[str], Awaitable[None]],
        cancelled: Callable[[], bool] | None = None,
        fallback_answer: str | None = None,
    ) -> tuple[str, str]:
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
            answer_parts: list[str] = []
            try:
                async with self.client_factory() as client:
                    async with client.stream(
                        "POST",
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model,
                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": context}],
                            "temperature": 0.2,
                            "max_tokens": settings.max_output_tokens,
                            "stream": True,
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if cancelled and cancelled():
                                break
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            payload = json.loads(data)
                            delta = payload["choices"][0].get("delta", {}).get("content")
                            if delta:
                                answer_parts.append(delta)
                                await on_token(delta)
                if answer_parts:
                    return "".join(answer_parts), f"{route_name}:{model}:stream"
            except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                if answer_parts:
                    return "".join(answer_parts), f"{route_name}:{model}:partial-stream"
                continue
        fallback = fallback_answer or self._grounded_fallback(hits)
        emitted: list[str] = []
        for index in range(0, len(fallback), 24):
            if cancelled and cancelled():
                break
            delta = fallback[index : index + 24]
            emitted.append(delta)
            await on_token(delta)
            await asyncio.sleep(0)
        return "".join(emitted), "local-grounded-fallback"

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
    def __init__(self, retriever: HybridRetriever | None = None, tool_model: Any | None = None) -> None:
        self.retriever = retriever or HybridRetriever()
        self.intent_router = IntentRouter()
        self.context_assembler = ContextAssembler(settings.context_token_budget)
        self.model_router = ModelRouter()
        self.tools = ToolRegistry()
        self.tools.register(
            "search_product_knowledge",
            "使用 BM25 与 Qdrant 向量召回、RRF 融合和 Rerank 检索产品知识，回答产品问题前应优先调用。",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的完整问题"},
                    "intent_code": {"type": "string", "description": "已识别的业务意图代码"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            self.retriever.search,
        )
        self.tools.register(
            "get_ingestion_task",
            "按任务 ID 查询文档入库状态、分块数量和错误信息。",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "32 位入库任务 ID"}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
            self._get_ingestion_task,
        )
        self.tools.register(
            "get_product_document",
            "根据检索结果中的文档 ID 获取产品文档元数据和内容摘要，用于核验来源。",
            {
                "type": "object",
                "properties": {"document_id": {"type": "string", "description": "产品文档 ID"}},
                "required": ["document_id"],
                "additionalProperties": False,
            },
            self._get_product_document,
        )
        self.react = ReActEngine(self.tools, tool_model or OpenAIToolCallingModel())

    @staticmethod
    def _get_ingestion_task(task_id: str) -> dict[str, Any]:
        with db.connect() as connection:
            row = connection.execute("SELECT * FROM ingestion_tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else {"error": "未找到入库任务"}

    @staticmethod
    def _get_product_document(document_id: str) -> dict[str, Any]:
        with db.connect() as connection:
            row = connection.execute(
                """SELECT d.id, d.doc_name, d.source_type, d.status, d.update_time,
                          COUNT(c.id) chunk_count, SUBSTR(d.raw_text, 1, 1200) excerpt
                   FROM documents d LEFT JOIN chunks c ON c.doc_id=d.id
                   WHERE d.id=? GROUP BY d.id""",
                (document_id,),
            ).fetchone()
        return dict(row) if row else {"error": "未找到产品文档"}

    @staticmethod
    def _hits_from_observations(observations: list[dict[str, Any]]) -> list[SearchHit]:
        for observation in observations:
            if observation["tool"] != "search_product_knowledge" or not isinstance(observation["result"], list):
                continue
            hits = []
            for item in observation["result"]:
                scores = item.get("scores") or {}
                hits.append(SearchHit(
                    chunk_id=item["chunkId"],
                    document_id=item["documentId"],
                    document_name=item["documentName"],
                    knowledge_base=item["knowledgeBase"],
                    content=item["content"],
                    score=float(item.get("score", 0)),
                    channel=item.get("channel", "bm25+qdrant+rrf+rerank"),
                    bm25_score=float(scores.get("bm25", 0)),
                    vector_score=float(scores.get("vector", 0)),
                    rerank_score=float(scores.get("rerank", 0)),
                ))
            return hits
        return []

    @staticmethod
    def _task_fallback(observations: list[dict[str, Any]]) -> str | None:
        task_observation = next(
            (item for item in observations if item["tool"] == "get_ingestion_task"),
            None,
        )
        if not task_observation:
            return None
        result = task_observation["result"]
        if result.get("error"):
            return result["error"]
        return (
            f"入库任务 {result.get('id')} 当前状态为 {result.get('status', '未知')}，"
            f"已生成 {result.get('chunk_count') or 0} 个分块。"
        )

    async def run(
        self,
        question: str,
        conversation_id: str,
        task_id: str,
        user: dict[str, Any],
        history: list[dict[str, str]],
        deep_thinking: bool = False,
        on_token: Callable[[str], Awaitable[None]] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AgentResult:
        trace = TraceRecorder(conversation_id, task_id, user)
        try:
            started = time.perf_counter()
            rewritten = self.retriever.rewrite(question, history) if settings.query_rewrite_enabled else question
            trace.node("问题改写", "rewrite", started, {"rewrittenQuestion": rewritten}, {"question": question})

            started = time.perf_counter()
            intent = self.intent_router.route(rewritten)
            trace.node("意图路由", "intent", started, intent, {"question": rewritten})

            recent_history = "\n".join(
                f"{item['role']}: {item['content'][:600]}" for item in history[-8:]
            ) or "无"
            system_prompt = (
                "你是企业 AI 产品智能助手，采用 ReAct 方式工作。根据问题自主选择注册工具；"
                "每次收到工具 Observation 后重新判断是否继续调用工具。产品事实必须先检索，"
                "不得编造价格、合同、客户隐私或未发布路线图。最终答案使用 [资料N] 标注证据。"
                f"最多允许 {settings.agent_max_steps} 轮规划。"
            )
            user_prompt = (
                f"[Original question]\n{question}\n\n[Rewritten question]\n{rewritten}\n\n"
                f"[Intent code]\n{intent['code']}\n\n[Intent]\n{intent['name']}\n\n"
                f"[Recent conversation]\n{recent_history}"
            )

            def record_react(step: int, phase: str, payload: dict[str, Any]) -> None:
                started = time.perf_counter()
                names = {
                    "plan": f"ReAct 第 {step} 轮规划",
                    "action": f"ReAct 第 {step} 轮 Action",
                    "observation": f"ReAct 第 {step} 轮 Observation",
                }
                name = names[phase]
                trace.node(name, phase, started, payload, {"step": step})

            react_result = await self.react.run(system_prompt, user_prompt, record_react, cancelled)
            hits = self._hits_from_observations(react_result.observations)
            context_started = time.perf_counter()
            context = self.context_assembler.build(
                question,
                rewritten,
                intent,
                history,
                hits,
                react_result.observations,
            )
            hits = context.selected_hits
            trace.node(
                "上下文筛选与预算组装",
                "context",
                context_started,
                {"tokenCount": context.token_count, **context.stats},
            )

            async def discard(_: str) -> None:
                return None

            fallback_answer = self._task_fallback(react_result.observations)
            generation_started = time.perf_counter()
            answer, route = await self.model_router.stream_complete(
                context.prompt,
                intent,
                hits,
                on_token or discard,
                cancelled,
                fallback_answer,
            )
            trace.node(
                "ReAct 最终回答",
                "generation",
                generation_started,
                {
                    "route": route,
                    "steps": react_result.steps,
                    "observations": len(react_result.observations),
                    "stoppedByLimit": react_result.stopped_by_limit,
                    "contextTokens": context.token_count,
                    "answerChars": len(answer),
                },
            )
            thinking = ""
            if deep_thinking:
                thinking = (
                    f"已将问题改写为“{rewritten}”，识别为“{intent['name']}”意图，"
                    f"Agent 完成 {react_result.steps} 轮规划并获得 {len(react_result.observations)} 次工具 Observation，"
                    f"最终基于 {len(hits)} 条重排证据生成回答。"
                )
            trace.finish()
            return AgentResult(answer, rewritten, intent, hits, route, trace.trace_id, task_id, thinking)
        except asyncio.CancelledError:
            trace.finish("cancelled", "生成任务已取消")
            raise
        except Exception as exc:
            trace.finish("error", str(exc)[:1000])
            raise


agent = ProductAssistantAgent()
