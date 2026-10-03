from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Protocol

import httpx

from .config import settings


def json_value(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable[..., Any],
    ) -> None:
        self._tools[name] = ToolDefinition(name, description, parameters, function)

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"工具 {name} 不存在")
        return self._tools[name].function(**arguments)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def descriptions(self) -> list[dict[str, Any]]:
        return [schema["function"] for schema in self.schemas()]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    content: str
    tool_calls: list[ToolCall]
    route: str


class ToolCallingModel(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn: ...


class OpenAIToolCallingModel:
    """OpenAI-compatible tool calling with a deterministic offline planner."""

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        providers: list[tuple[str, str, str, str]] = []
        if settings.llm_api_key:
            providers.append((settings.llm_base_url, settings.llm_api_key, settings.llm_model, "primary"))
        if settings.secondary_api_key and settings.secondary_base_url and settings.secondary_model:
            providers.append(
                (settings.secondary_base_url, settings.secondary_api_key, settings.secondary_model, "secondary")
            )
        for base_url, api_key, model, route in providers:
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model,
                            "messages": messages,
                            "tools": tools,
                            "tool_choice": "auto",
                            "temperature": 0.2,
                        },
                    )
                    response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                calls = []
                for raw in message.get("tool_calls") or []:
                    arguments = raw["function"].get("arguments") or "{}"
                    calls.append(ToolCall(
                        id=raw.get("id") or uuid.uuid4().hex,
                        name=raw["function"]["name"],
                        arguments=json.loads(arguments) if isinstance(arguments, str) else arguments,
                    ))
                return ModelTurn(message.get("content") or "", calls, f"{route}:{model}")
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return self._local_plan(messages)

    @staticmethod
    def _local_plan(messages: list[dict[str, Any]]) -> ModelTurn:
        user_question = next(
            (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        observations = [message for message in messages if message.get("role") == "tool"]
        if not observations:
            task_match = re.search(r"\b[0-9a-f]{32}\b", user_question, flags=re.IGNORECASE)
            if task_match and re.search(r"(任务|入库|状态|进度)", user_question):
                return ModelTurn("", [ToolCall(uuid.uuid4().hex, "get_ingestion_task", {"task_id": task_match.group()})], "local-react-planner")
            query_match = re.search(r"\[Rewritten question\]\n(.+)", user_question)
            query = query_match.group(1).strip() if query_match else user_question
            intent_match = re.search(r"\[Intent code\]\n([^\n]+)", user_question)
            arguments: dict[str, Any] = {"query": query}
            if intent_match:
                arguments["intent_code"] = intent_match.group(1).strip()
            return ModelTurn("", [ToolCall(uuid.uuid4().hex, "search_product_knowledge", arguments)], "local-react-planner")

        last = observations[-1]
        if last.get("name") == "search_product_knowledge":
            try:
                hits = json.loads(str(last.get("content", "[]")))
            except json.JSONDecodeError:
                hits = []
            if hits and isinstance(hits, list) and hits[0].get("documentId"):
                return ModelTurn(
                    "",
                    [ToolCall(uuid.uuid4().hex, "get_product_document", {"document_id": hits[0]["documentId"]})],
                    "local-react-planner",
                )
        return ModelTurn("__LOCAL_FINAL__", [], "local-react-planner")


@dataclass
class ReActResult:
    answer: str
    route: str
    steps: int
    observations: list[dict[str, Any]]
    stopped_by_limit: bool = False


TraceCallback = Callable[[int, str, dict[str, Any]], None]


class ReActEngine:
    def __init__(
        self,
        registry: ToolRegistry,
        model: ToolCallingModel,
        max_steps: int | None = None,
    ) -> None:
        self.registry = registry
        self.model = model
        self.max_steps = max_steps or settings.agent_max_steps

    async def run(
        self,
        system_prompt: str,
        user_prompt: str,
        trace: TraceCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ReActResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        observations: list[dict[str, Any]] = []
        route = "unknown"
        for step in range(1, self.max_steps + 1):
            if cancelled and cancelled():
                raise asyncio.CancelledError
            turn = await self.model.complete(messages, self.registry.schemas())
            if cancelled and cancelled():
                raise asyncio.CancelledError
            route = turn.route
            if trace:
                trace(step, "plan", {
                    "route": route,
                    "selectedTools": [call.name for call in turn.tool_calls],
                    "hasFinalAnswer": bool(turn.content and turn.content != "__LOCAL_FINAL__"),
                })
            if not turn.tool_calls:
                return ReActResult(turn.content, route, step, observations)

            assistant_calls = []
            for call in turn.tool_calls:
                assistant_calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                })
            messages.append({"role": "assistant", "content": turn.content or None, "tool_calls": assistant_calls})

            for call in turn.tool_calls:
                if cancelled and cancelled():
                    raise asyncio.CancelledError
                if trace:
                    trace(step, "action", {"tool": call.name, "arguments": call.arguments})
                try:
                    result = json_value(await asyncio.to_thread(self.registry.execute, call.name, call.arguments))
                except Exception as exc:
                    result = {"error": str(exc)}
                if cancelled and cancelled():
                    raise asyncio.CancelledError
                observation = {"tool": call.name, "arguments": call.arguments, "result": result}
                observations.append(observation)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                if trace:
                    trace(step, "observation", observation)

        return ReActResult("", route, self.max_steps, observations, stopped_by_limit=True)
